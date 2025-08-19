import os
import json
import hashlib
import time
import glob
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import networkx as nx
from dataclasses import dataclass
import logging
from pathlib import Path
import uuid # For temporary file names
import subprocess # For calling external commands
from subprocess import Popen, PIPE, STDOUT # Explicit imports for subprocess

# RDF/OWL processing
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from rdflib.exceptions import ParserError # Import specific ParserError

# For NLTK WordNet (lexical similarity)
try:
    import nltk
    from nltk.corpus import wordnet
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logging.warning("NLTK (Natural Language Toolkit) not found. Lexical similarity will be limited.")
    logging.warning("Please install with: pip install nltk, then run: import nltk; nltk.download('wordnet')")

# Owlready2 is no longer a hard dependency for reasoner execution in this version,
# as we're using subprocess. Keep the import for potential future Owlready2-specific
# parsing or manipulations, but it's not used for reasoner execution in _validate_unified_ontology.
_OWLREADY2_AVAILABLE = False
try:
    import owlready2
    _OWLREADY2_AVAILABLE = True
    # For future, if specific Owlready2 API needed for parsing etc.
    # from owlready2 import get_ontology, default_world, sync_reasoner
except ImportError:
    logging.info("Owlready2 is not installed, so its specific API capabilities are unavailable.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scw_algorithm.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConceptEntry:
    """Data class for concept entries"""
    ontology_id: str
    concept: Dict[str, Any]
    signature: str
    relationships: List[Dict[str, str]]
    properties: Dict[str, Any] # Changed type hint to Any as it holds dicts from _get_concept_properties_for_signature

@dataclass
class UnifiedConcept:
    """Data class for unified concepts"""
    concept_id: str
    type: str  # CROSS_ONTOLOGY, UNIQUE_CONCEPT, PARTIAL_CROSS_ONTOLOGY
    original_label: str
    source_entries: List[ConceptEntry]
    match_type: str
    coverage: str # <--- THIS WAS THE MISSING FIELD
    coverage_percentage: float
    contexts: Dict[str, Dict[str, Any]]
    missing_ontologies: Optional[List[str]] = None

class SCWProcessor:
    """
    Main Semantic Context Weaving Processor
    Handles ANY number of ontologies (2 to 1000+)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.scw_namespace = Namespace("http://scw.org/unified#")
        
        # Default configuration values
        self.config = {
            'similarity_rel_weight': 0.4, # Weight for relationship similarity
            'similarity_prop_weight': 0.25, # Weight for property similarity
            'similarity_sig_weight': 0.2, # Weight for signature similarity
            'similarity_lex_weight': 0.15, # Weight for lexical similarity (new)
            'similarity_threshold': 0.3,
            'bridge_threshold': 0.35,
            'max_bridges_per_concept': 5,
            'enable_compensation_bridges': True,
            'processing_mode': 'memory', # 'memory', 'streaming' (conceptual), 'batch' (conceptual)
            'batch_size': None,
            'enable_reasoning_validation': False # New: Enable reasoner for final validation
        }
        if config:
            self.config.update(config)
        
        # Processing data
        self.original_ontologies = {} # Stores successfully parsed ontologies
        self.unparsable_files = [] # New: To track files that failed to parse
        self.concept_catalog = defaultdict(list) # Key: normalized_label -> List[ConceptEntry]
        self.unified_concepts = {} # Key: unified_id -> UnifiedConcept data
        self.cross_relationships = []
        self.bridge_relationships = []
        self.compensation_bridges = []
        
        # Mapping for quick lookup (new for explicit equivalences and properties)
        self._uri_to_concept_entry = {} # original_uri -> ConceptEntry

        # Statistics
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_input_files_attempted': 0, # New: Count of all files user provided
            'total_input_concepts': 0,
            'total_input_relationships': 0,
            'total_unified_concepts': 0,
            'cross_ontology_matches': 0,
            'unique_concepts': 0,
            'partial_matches': 0,
            'explicit_equivalences_found': 0 # New stat
        }

    def execute(self, ontology_files: List[str], output_file: str, 
                config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main execution method - handles ANY number of ontologies
        
        Args:
            ontology_files: List of OWL file paths (2 to unlimited)
            output_file: Output unified OWL file path
            config: Optional configuration parameters
            
        Returns:
            Execution result dictionary
        """
        
        self.processing_stats['start_time'] = time.time()
        self.processing_stats['total_input_files_attempted'] = len(ontology_files)
        
        try:
            # Auto-configure based on scale (based on files attempted, but logic uses actually loaded)
            self._auto_configure(len(ontology_files), config) 
            
            logger.info(f"Starting SCW Algorithm with {len(ontology_files)} input files...")
            logger.info(f"Configuration: {json.dumps(self.config, indent=2)}") 
            
            # Phase 1: Load and analyze (now includes robust parsing)
            self._phase1_load_and_analyze(ontology_files)

            # Validate that enough ontologies were successfully parsed
            if len(self.original_ontologies) < 2:
                error_msg = f'Only {len(self.original_ontologies)} ontology files were successfully parsed. At least 2 valid ontology files are required for meaningful unification. Unparsable files: {self.unparsable_files}'
                logger.error(f"SCW Algorithm failed: {error_msg}")
                return {
                    'status': 'error',
                    'error': error_msg,
                    'processing_stats': self.processing_stats,
                    'unparsable_files': self.unparsable_files
                }
            
            # Re-run auto-configure if necessary based on actual loaded ontologies (optional, but robust)
            if len(self.original_ontologies) != self.processing_stats['total_input_files_attempted']:
                 self._auto_configure(len(self.original_ontologies), config)
                 logger.info(f"Re-auto-configured based on {len(self.original_ontologies)} successfully loaded ontologies.")


            # Execute remaining processing phases
            self._phase2_create_concept_catalog()
            matches, unique_concepts, partial_matches = self._phase3_detect_matches()
            self._phase4_create_unified_namespace(matches, unique_concepts, partial_matches)
            self._phase5_discover_cross_relationships()
            self._phase6_create_bridges_and_compensation()
            unified_owl = self._phase7_generate_unified_owl()
            validation_report = self._phase8_save_and_validate(unified_owl, output_file)
            
            self.processing_stats['end_time'] = time.time()
            
            logger.info("SCW Algorithm completed successfully!")
            
            return {
                'status': 'success',
                'output_file': output_file,
                'validation_report': validation_report,
                'processing_stats': self._finalize_stats(),
                'config_used': self.config,
                'unparsable_files': self.unparsable_files # Report unparsable files
            }
            
        except Exception as e:
            logger.error(f"SCW Algorithm failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'processing_stats': self.processing_stats,
                'unparsable_files': self.unparsable_files
            }

    def _validate_inputs(self, ontology_files: List[str]) -> Dict[str, Any]:
        """Validate input files and parameters (basic existence and count checks)."""
        
        # This function now only does basic checks before parsing.
        # The actual parsing validation happens in _phase1_load_and_analyze.

        if len(ontology_files) < 2:
            return {
                'valid': False,
                'error': 'At least 2 ontology file paths must be provided to the algorithm.'
            }
        
        missing_files = []
        for file_path in ontology_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            return {
                'valid': False,
                'error': f'The following ontology files were not found: {missing_files}'
            }
        
        return {'valid': True}

    def _auto_configure(self, num_ontologies: int, user_config: Optional[Dict] = None):
        """Auto-configure algorithm parameters based on scale"""
        
        # Base configuration based on number of ontologies
        # Weights for similarity are adjusted based on scale as well
        if num_ontologies <= 5:
            base_config = {
                'scale': 'SMALL',
                'similarity_threshold': 0.2,
                'bridge_threshold': 0.25,
                'max_bridges_per_concept': 8,
                'enable_compensation_bridges': True,
                'processing_mode': 'memory',
                'similarity_rel_weight': 0.5, 'similarity_prop_weight': 0.25,
                'similarity_sig_weight': 0.15, 'similarity_lex_weight': 0.1
            }
        elif num_ontologies <= 25:
            base_config = {
                'scale': 'MEDIUM',
                'similarity_threshold': 0.3,
                'bridge_threshold': 0.3,
                'max_bridges_per_concept': 10,
                'enable_compensation_bridges': True,
                'processing_mode': 'memory',
                'similarity_rel_weight': 0.45, 'similarity_prop_weight': 0.25,
                'similarity_sig_weight': 0.2, 'similarity_lex_weight': 0.1
            }
        elif num_ontologies <= 100:
            base_config = {
                'scale': 'LARGE',
                'similarity_threshold': 0.4,
                'bridge_threshold': 0.35,
                'max_bridges_per_concept': 12,
                'enable_compensation_bridges': True,
                'processing_mode': 'memory', # For now, keep as memory. True streaming needs architectural changes.
                'similarity_rel_weight': 0.4, 'similarity_prop_weight': 0.2,
                'similarity_sig_weight': 0.25, 'similarity_lex_weight': 0.15
            }
        else: # num_ontologies > 100
            base_config = {
                'scale': 'MASSIVE',
                'similarity_threshold': 0.5,
                'bridge_threshold': 0.4,
                'max_bridges_per_concept': 15,
                'enable_compensation_bridges': True,
                'processing_mode': 'memory', # For true 'streaming' or 'batch' for massive, consider a persistent triple store like Fuseki/Blazegraph or a custom on-disk RDFLib Store.
                'batch_size': 1000, # Conceptual batch size for future streaming
                'similarity_rel_weight': 0.35, 'similarity_prop_weight': 0.15,
                'similarity_sig_weight': 0.3, 'similarity_lex_weight': 0.2
            }
        
        # Merge with user configuration
        self.config.update(base_config) # Start with base, then update with user
        if user_config:
            self.config.update(user_config)
        
        logger.info(f"Auto-configured for {self.config['scale']} scale processing")

    def _phase1_load_and_analyze(self, ontology_files: List[str]):
        """
        Phase 1: Load and analyze all ontology files.
        Includes robust error handling for unparsable files.
        """
        
        logger.info(f"Phase 1: Attempting to load {len(ontology_files)} ontologies...")
        
        total_concepts = 0
        total_relationships = 0
        self.unparsable_files = [] # Reset for this run
        
        for i, file_path in enumerate(ontology_files):
            try:
                ontology_id = Path(file_path).stem
                
                # Handle duplicate ontology IDs
                original_id = ontology_id
                counter = 1
                while ontology_id in self.original_ontologies:
                    ontology_id = f"{original_id}_{counter}"
                    counter += 1
                
                logger.info(f"Loading ontology {i+1}/{len(ontology_files)}: {ontology_id} from {file_path}")
                
                # Parse OWL file - Robustness added here!
                graph = Graph()
                try:
                    graph.parse(file_path) # RDFLib automatically detects format for many common ones
                except ParserError as e:
                    logger.error(f"Skipping {file_path} due to parsing error: {str(e)}")
                    self.unparsable_files.append({'file_path': file_path, 'error': str(e)})
                    continue # Skip to the next file if parsing fails
                except Exception as e: # Catch other potential parsing exceptions
                    logger.error(f"Skipping {file_path} due to unexpected parsing error: {str(e)}")
                    self.unparsable_files.append({'file_path': file_path, 'error': str(e)})
                    continue

                # Extract concepts, relationships, and explicit equivalences
                concepts = self._extract_concepts(graph)
                relationships, properties_meta = self._extract_relationships(graph) # Modified to get property metadata
                explicit_equivalences = self._extract_explicit_equivalences(graph) # New extraction
                
                self.original_ontologies[ontology_id] = {
                    'file_path': file_path,
                    'graph': graph, # Keep graph for reference, but concepts/rels are primary
                    'concepts': concepts,
                    'relationships': relationships,
                    'properties_meta': properties_meta, # Store property metadata
                    'explicit_equivalences': explicit_equivalences, # Store explicit equivalences
                    'namespaces': dict(graph.namespaces()),
                    'stats': {
                        'concept_count': len(concepts),
                        'relationship_count': len(relationships),
                        'explicit_eq_count': len(explicit_equivalences)
                    }
                }
                
                total_concepts += len(concepts)
                total_relationships += len(relationships)
                self.processing_stats['explicit_equivalences_found'] += len(explicit_equivalences)
                
                # Populate URI to ConceptEntry map
                for concept_data in concepts:
                    self._uri_to_concept_entry[concept_data['uri']] = ConceptEntry(
                        ontology_id=ontology_id,
                        concept=concept_data,
                        signature="", # Will be filled in Phase 2
                        relationships=[], # Will be filled in Phase 2
                        properties=concept_data.get('properties', {})
                    )
                
                logger.info(f"  Loaded: {len(concepts)} concepts, {len(relationships)} relationships, {len(explicit_equivalences)} explicit equivalences")
                
            except Exception as e:
                # This outer catch is for issues *after* successful parsing, or unexpected errors
                logger.error(f"An error occurred while processing {file_path}: {str(e)}", exc_info=True)
                self.unparsable_files.append({'file_path': file_path, 'error': str(e)})
        
        self.processing_stats['total_input_concepts'] = total_concepts
        self.processing_stats['total_input_relationships'] = total_relationships
        
        logger.info(f"Phase 1 complete: Successfully loaded {len(self.original_ontologies)} ontologies. Skipped {len(self.unparsable_files)} unparsable files.")
        logger.info(f"{total_concepts} total concepts, {total_relationships} total relationships, {self.processing_stats['explicit_equivalences_found']} explicit equivalences found in loaded files.")


    def _extract_concepts(self, graph: Graph) -> List[Dict[str, Any]]:
        """Extract all concepts (OWL classes) from a graph"""
        
        concepts = []
        
        for subject in graph.subjects(RDF.type, OWL.Class):
            if isinstance(subject, URIRef):
                # Skip built-in OWL classes
                if str(subject).startswith(str(OWL)):
                    continue
                
                concept = {
                    'uri': str(subject),
                    'local_name': self._extract_local_name(str(subject)),
                    'properties': {}, # Data properties related to this class
                    'annotations': {} # rdfs:comment, owl:versionInfo, etc.
                }
                
                # Extract rdfs:label
                labels = list(graph.objects(subject, RDFS.label))
                concept['label'] = str(labels[0]) if labels else concept['local_name']
                
                # Extract rdfs:comment and other annotations
                for p, o in graph.predicate_objects(subject):
                    if str(p).startswith(str(RDFS)) or str(p).startswith(str(OWL)) and p != RDF.type:
                        if p == RDFS.comment:
                            concept['annotations']['comment'] = str(o)
                        elif p == OWL.versionInfo:
                            concept['annotations']['versionInfo'] = str(o)
                        # Add more annotations if needed
                    elif isinstance(o, Literal): # Direct data properties
                        prop_name = self._extract_local_name(str(p))
                        concept['properties'][prop_name] = str(o)
                
                concepts.append(concept)
        
        return concepts

    def _extract_relationships(self, graph: Graph) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
        """
        Extract relationships (object properties) and gather metadata for all properties.
        Returns: (list of relationships, dict of property metadata)
        """
        
        relationships = []
        properties_meta = defaultdict(lambda: {'type': 'unknown', 'characteristics': []})
        
        # Identify ObjectProperties and DataProperties and their characteristics
        for prop_uri in graph.subjects(RDF.type, OWL.ObjectProperty):
            prop_str = str(prop_uri)
            properties_meta[prop_str]['type'] = 'object'
            for p, o in graph.predicate_objects(prop_uri):
                if p == OWL.inverseOf:
                    properties_meta[prop_str]['characteristics'].append(f"inverseOf:{self._extract_local_name(str(o))}")
                elif p == RDF.type:
                    if o == OWL.TransitiveProperty: properties_meta[prop_str]['characteristics'].append('transitive')
                    if o == OWL.SymmetricProperty: properties_meta[prop_str]['characteristics'].append('symmetric')
                    if o == OWL.AsymmetricProperty: properties_meta[prop_str]['characteristics'].append('asymmetric')
                    if o == OWL.ReflexiveProperty: properties_meta[prop_str]['characteristics'].append('reflexive')
                    if o == OWL.IrreflexiveProperty: properties_meta[prop_str]['characteristics'].append('irreflexive')
                    if o == OWL.FunctionalProperty: properties_meta[prop_str]['characteristics'].append('functional')
                    if o == OWL.InverseFunctionalProperty: properties_meta[prop_str]['characteristics'].append('inverseFunctional')
                elif p == RDFS.label:
                     properties_meta[prop_str]['label'] = str(o)

        for prop_uri in graph.subjects(RDF.type, OWL.DatatypeProperty):
            prop_str = str(prop_uri)
            properties_meta[prop_str]['type'] = 'data'
            for p, o in graph.predicate_objects(prop_uri):
                 if p == RDFS.label:
                     properties_meta[prop_str]['label'] = str(o)
                 if o == OWL.FunctionalProperty: properties_meta[prop_str]['characteristics'].append('functional')


        # Extract triples that represent object properties between URIRefs
        for subject, predicate, obj in graph:
            # Only include relationships between URIRefs (concepts/individuals)
            # and exclude OWL internal predicates
            if (isinstance(subject, URIRef) and isinstance(obj, URIRef) and
                predicate not in [RDF.type, RDFS.label, RDFS.comment, OWL.equivalentClass, OWL.sameAs] and
                not str(predicate).startswith(str(OWL))): # Exclude all OWL predicates
                
                relationship = {
                    'subject': str(subject),
                    'predicate': str(predicate),
                    'object': str(obj),
                    'predicate_local': self._extract_local_name(str(predicate)),
                    'subject_local': self._extract_local_name(str(subject)),
                    'object_local': self._extract_local_name(str(obj))
                }
                relationships.append(relationship)
        
        return relationships, dict(properties_meta)

    def _extract_explicit_equivalences(self, graph: Graph) -> List[Tuple[str, str]]:
        """Extract owl:equivalentClass and owl:sameAs axioms for high-confidence matches."""
        equivalences = []
        # owl:equivalentClass
        for s, o in graph.subject_objects(OWL.equivalentClass):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                equivalences.append((str(s), str(o)))
                equivalences.append((str(o), str(s))) # Bidirectional
        
        # owl:sameAs
        for s, o in graph.subject_objects(OWL.sameAs):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                equivalences.append((str(s), str(o)))
                equivalences.append((str(o), str(s))) # Bidirectional
        return list(set(equivalences)) # Deduplicate

    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI"""
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        else:
            return uri

    def _get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word using WordNet (if NLTK available)."""
        if not _NLTK_AVAILABLE:
            return {word.lower()} # Fallback to just the word itself

        synonyms = set()
        # Handle multi-word labels by splitting and getting synonyms for each part
        words = word.lower().replace('-', ' ').split()
        for w in words:
            for syn in wordnet.synsets(w):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower().replace('_', ' ')) # Normalize
        
        # Add original words too
        synonyms.update(words)
        return synonyms

    def _phase2_create_concept_catalog(self):
        """Phase 2: Create global concept catalog with signatures"""
        
        logger.info("Phase 2: Creating comprehensive concept catalog...")
        
        for ontology_id, ontology_data in self.original_ontologies.items():
            logger.info(f"  Cataloging concepts from {ontology_id}...")
            
            for concept in ontology_data['concepts']:
                # Normalize concept label for matching
                concept_label = concept['label'].lower().strip()
                
                # Get relationships for this concept
                concept_relationships = self._get_concept_relationships(
                    concept['uri'], ontology_data['relationships']
                )
                
                # Get relevant properties for signature (including characteristics)
                concept_properties_for_signature = self._get_concept_properties_for_signature(
                    concept['uri'], ontology_data['graph'], ontology_data['properties_meta']
                )

                # Create semantic signature
                signature = self._create_concept_signature(
                    concept, concept_relationships, concept_properties_for_signature
                )
                
                # Update the ConceptEntry in our mapping
                entry = self._uri_to_concept_entry[concept['uri']]
                entry.signature = signature
                entry.relationships = concept_relationships
                entry.properties = concept_properties_for_signature # Use enriched properties

                self.concept_catalog[concept_label].append(entry)
        
        logger.info(f"Phase 2 complete: Cataloged {len(self.concept_catalog)} unique concept labels")

    def _create_concept_signature(self, concept: Dict[str, Any], 
                                  relationships: List[Dict[str, str]],
                                  properties_for_signature: Dict[str, Any]) -> str:
        """Create semantic fingerprint for concept, including lexical hints."""
        
        # Extract signature components
        predicates = sorted(set(rel['predicate_local'] for rel in relationships))
        outgoing_connections = len(set(rel['object'] for rel in relationships if rel['subject'] == concept['uri']))
        incoming_connections = len(set(rel['subject'] for rel in relationships 
                                   if rel['object'] == concept['uri']))
        
        # Use property local names and their characteristics for signature
        prop_keys_and_chars = []
        for prop_uri, prop_data in properties_for_signature.items():
            prop_keys_and_chars.append(self._extract_local_name(prop_uri))
            prop_keys_and_chars.extend(prop_data.get('meta', {}).get('characteristics', []))
        prop_keys_and_chars = sorted(set(prop_keys_and_chars))

        # Add lexical components
        concept_label = concept.get('label', concept.get('local_name', ''))
        lexical_hints = sorted(list(self._get_synonyms(concept_label)))

        # Create signature data
        signature_data = {
            'predicates': predicates[:10],  # Limit to avoid huge signatures
            'outgoing_connections': outgoing_connections,
            'incoming_connections': incoming_connections,
            'properties_fingerprint': prop_keys_and_chars[:10], # Use combined property info
            'lexical_hints': lexical_hints[:5], # Limit lexical hints
            'concept_complexity': len(predicates) + len(properties_for_signature)
        }
        
        # Generate hash
        signature_string = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]

    def _get_concept_relationships(self, concept_uri: str, 
                                 relationships: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Get all relationships where concept appears as subject or object"""
        
        concept_rels = []
        for rel in relationships:
            if rel['subject'] == concept_uri or rel['object'] == concept_uri:
                concept_rels.append(rel)
        
        return concept_rels

    def _get_concept_properties_for_signature(self, concept_uri: str, graph: Graph, 
                                              properties_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Get properties directly associated with a concept URI,
        including their characteristics for the signature.
        """
        concept_props = {}
        for p, o in graph.predicate_objects(URIRef(concept_uri)):
            # Only consider direct properties (not rdfs:label, rdf:type, etc.)
            if isinstance(p, URIRef) and isinstance(o, Literal) and \
               p not in [RDF.type, RDFS.label, RDFS.comment]:
                prop_uri_str = str(p)
                concept_props[prop_uri_str] = {
                    'value': str(o),
                    'meta': properties_meta.get(prop_uri_str, {'type': 'data', 'characteristics': []})
                }
            # Also include object properties where this concept is subject or object
            # (only if they are explicitly declared properties in properties_meta)
            elif isinstance(p, URIRef) and isinstance(o, URIRef) and \
                 p not in [RDF.type, RDFS.label, RDFS.comment]:
                prop_uri_str = str(p)
                if properties_meta.get(prop_uri_str, {}).get('type') == 'object':
                     concept_props[prop_uri_str] = {
                        'value': str(o), # Store target URI here
                        'meta': properties_meta.get(prop_uri_str, {'type': 'object', 'characteristics': []})
                    }
        return concept_props


    def _phase3_detect_matches(self) -> Tuple[Dict, Dict, Dict]:
        """Phase 3: Detect cross-ontology matches with dynamic scaling and explicit equivalences"""
        
        logger.info("Phase 3: Detecting cross-ontology matches...")
        
        total_ontologies = len(self.original_ontologies) # Only count successfully loaded ontologies
        full_matches = {}      # Concepts in ALL loaded ontologies
        partial_matches = {}   # Concepts in SOME loaded ontologies (2 to N-1)
        unique_concepts = {}   # Concepts in only ONE loaded ontology
        
        processed_uris = set() # To prevent double processing concepts due to explicit equivalences

        # --- New: Prioritize Explicit Equivalences ---
        # This section requires careful design to 'pre-merge' concepts before general similarity.
        # For a truly robust implementation, you would:
        # 1. Build a graph of explicit equivalences across ALL *successfully loaded* ontologies.
        # 2. Find connected components in this graph.
        # 3. Each component represents a single 'pre-unified' concept that *must* be merged.
        # 4. These 'pre-unified' concepts would then bypass the standard similarity matching,
        #    and their constituent original ConceptEntries would be marked as 'processed_uris'.
        # For now, we'll just log them and update a stat, but they are not yet fully integrated
        # into the auto-merging process, relying on label/semantic similarity for unification.
        
        # Example of how to iterate and potentially use them (no auto-merge here yet)
        for ont_id, ont_data in self.original_ontologies.items():
            for s_uri, o_uri in ont_data.get('explicit_equivalences', []):
                s_entry = self._uri_to_concept_entry.get(s_uri)
                o_entry = self._uri_to_concept_entry.get(o_uri)
                
                if s_entry and o_entry and s_entry.ontology_id != o_entry.ontology_id:
                    # Log for now; complex auto-merging logic would go here
                    # e.g., create a new UnifiedConcept directly from these two entries,
                    # and remove them from self.concept_catalog or mark their URIs as processed_uris
                    logger.debug(f"Explicit equivalence found between {s_entry.ontology_id}:{s_entry.concept['label']} and {o_entry.ontology_id}:{o_entry.concept['label']}")
                    # If you implement auto-merge for explicit, add to processed_uris:
                    # processed_uris.add(s_uri)
                    # processed_uris.add(o_uri)
        # --- End Explicit Equivalences ---


        for concept_label, entries in self.concept_catalog.items():
            # If explicit merging was implemented above, this would skip already merged concepts
            if any(entry.concept['uri'] in processed_uris for entry in entries):
                continue

            # Ensure we only consider entries from successfully loaded ontologies
            relevant_entries = [entry for entry in entries if entry.ontology_id in self.original_ontologies]
            if not relevant_entries: # If all entries for this label were from unparsable files
                continue

            num_ontologies_with_concept = len(set(entry.ontology_id for entry in relevant_entries)) 
            coverage_percentage = (num_ontologies_with_concept / total_ontologies) * 100
            
            if num_ontologies_with_concept == 1:
                # Unique to one ontology
                unique_concepts[concept_label] = relevant_entries[0]
                
            elif num_ontologies_with_concept == total_ontologies:
                # Appears in ALL loaded ontologies
                similarity_matrix = self._calculate_similarity_matrix(relevant_entries)
                match_type = self._determine_match_type(similarity_matrix, "FULL")
                
                full_matches[concept_label] = {
                    'entries': relevant_entries,
                    'similarity_matrix': similarity_matrix,
                    'match_type': match_type,
                    'coverage': 'FULL_COVERAGE',
                    'coverage_percentage': 100.0,
                    'ontology_count': num_ontologies_with_concept
                }
                
            else:
                # Appears in SOME loaded ontologies (partial coverage)
                similarity_matrix = self._calculate_similarity_matrix(relevant_entries)
                match_type = self._determine_match_type(similarity_matrix, "PARTIAL")
                
                partial_matches[concept_label] = {
                    'entries': relevant_entries,
                    'similarity_matrix': similarity_matrix,
                    'match_type': match_type,
                    'coverage': 'PARTIAL_COVERAGE',
                    'coverage_percentage': coverage_percentage,
                    'ontology_count': num_ontologies_with_concept,
                    'missing_ontologies': self._find_missing_ontologies_for_concept(relevant_entries)
                }
        
        # Update statistics
        self.processing_stats['cross_ontology_matches'] = len(full_matches)
        self.processing_stats['partial_matches'] = len(partial_matches)
        self.processing_stats['unique_concepts'] = len(unique_concepts)
        
        logger.info(f"Phase 3 complete:")
        logger.info(f"  - Full cross-ontology matches: {len(full_matches)}")
        logger.info(f"  - Partial cross-ontology matches: {len(partial_matches)}")
        logger.info(f"  - Unique concepts: {len(unique_concepts)}")
        logger.info(f"  - Explicit equivalences found (conceptual merge): {self.processing_stats['explicit_equivalences_found']}")
        
        return full_matches, unique_concepts, partial_matches

    def _calculate_similarity_matrix(self, entries: List[ConceptEntry]) -> Dict[str, float]:
        """Calculate pairwise similarity between concept entries using configurable weights."""
        
        similarity_matrix = {}
        
        rel_weight = self.config.get('similarity_rel_weight', 0.4)
        prop_weight = self.config.get('similarity_prop_weight', 0.25)
        sig_weight = self.config.get('similarity_sig_weight', 0.2)
        lex_weight = self.config.get('similarity_lex_weight', 0.15)
        
        # Ensure weights sum to 1, if not, normalize or log warning
        total_weight = rel_weight + prop_weight + sig_weight + lex_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Similarity weights do not sum to 1.0 ({total_weight}). Normalizing.")
            rel_weight /= total_weight
            prop_weight /= total_weight
            sig_weight /= total_weight
            lex_weight /= total_weight

        for i, entry1 in enumerate(entries):
            for j, entry2 in enumerate(entries):
                if i < j:  # Avoid duplicate calculations
                    # Multiple similarity metrics
                    rel_sim = self._calculate_relationship_similarity(
                        entry1.relationships, entry2.relationships
                    )
                    prop_sim = self._calculate_property_similarity(
                        entry1.properties, entry2.properties
                    )
                    sig_sim = 1.0 if entry1.signature == entry2.signature else 0.0
                    
                    # Lexical similarity
                    label1 = entry1.concept.get('label', entry1.concept.get('local_name', ''))
                    label2 = entry2.concept.get('label', entry2.concept.get('local_name', ''))
                    
                    terms1 = self._get_synonyms(label1)
                    terms2 = self._get_synonyms(label2)
                    lexical_sim = self._jaccard_similarity(terms1, terms2)
                    
                    # Weighted combination
                    overall_similarity = (rel_sim * rel_weight) + \
                                         (prop_sim * prop_weight) + \
                                         (sig_sim * sig_weight) + \
                                         (lexical_sim * lex_weight)
                    
                    similarity_matrix[f"{i}-{j}"] = overall_similarity
        
        return similarity_matrix

    def _calculate_relationship_similarity(self, rels1: List[Dict], rels2: List[Dict]) -> float:
        """Calculate Jaccard similarity for relationship sets (based on predicate local names)."""
        
        if not rels1 and not rels2:
            return 1.0
        if not rels1 or not rels2:
            return 0.0
        
        # Using predicate_local for comparison
        predicates1 = set(rel['predicate_local'] for rel in rels1)
        predicates2 = set(rel['predicate_local'] for rel in rels2)
        
        return self._jaccard_similarity(predicates1, predicates2)

    def _calculate_property_similarity(self, props1: Dict, props2: Dict) -> float:
        """Calculate Jaccard similarity for property sets (based on property URIs and characteristics)."""
        
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.0
        
        # Create a set of "property fingerprints" including local name and characteristics
        fingerprints1 = set()
        for uri, data in props1.items():
            fingerprint = self._extract_local_name(uri)
            fingerprint += "_" + "_".join(sorted(data.get('meta', {}).get('characteristics', [])))
            fingerprints1.add(fingerprint)

        fingerprints2 = set()
        for uri, data in props2.items():
            fingerprint = self._extract_local_name(uri)
            fingerprint += "_" + "_".join(sorted(data.get('meta', {}).get('characteristics', [])))
            fingerprints2.add(fingerprint)
        
        return self._jaccard_similarity(fingerprints1, fingerprints2)

    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity coefficient"""
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def _determine_match_type(self, similarity_matrix: Dict[str, float], 
                            coverage_type: str) -> str:
        """Determine match type with dynamic thresholds"""
        
        if not similarity_matrix:
            return f"SINGLE_{coverage_type}" # Or a specific tag for this scenario
        
        similarities = list(similarity_matrix.values())
        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        
        # Dynamic thresholds based on scale (can be further fine-tuned)
        scale = self.config.get('scale', 'MEDIUM')
        
        if scale == 'SMALL':
            exact_thresh, strong_thresh, moderate_thresh = 0.85, 0.65, 0.45
        elif scale == 'MEDIUM':
            exact_thresh, strong_thresh, moderate_thresh = 0.90, 0.70, 0.50
        elif scale == 'LARGE':
            exact_thresh, strong_thresh, moderate_thresh = 0.92, 0.75, 0.55
        else:  # MASSIVE
            exact_thresh, strong_thresh, moderate_thresh = 0.95, 0.80, 0.60
        
        # Classify match intensity
        if avg_similarity >= exact_thresh and min_similarity >= 0.8:
            intensity = "EXACT"
        elif avg_similarity >= strong_thresh and min_similarity >= 0.5:
            intensity = "STRONG"
        elif avg_similarity >= moderate_thresh:
            intensity = "MODERATE"
        else:
            intensity = "WEAK"
        
        return f"{intensity}_{coverage_type}_MATCH"

    def _find_missing_ontologies_for_concept(self, entries: List[ConceptEntry]) -> List[str]:
        """Find ontologies where this concept doesn't appear"""
        
        present_ontologies = set(entry.ontology_id for entry in entries)
        all_ontologies = set(self.original_ontologies.keys()) # Only from successfully parsed
        
        return list(all_ontologies - present_ontologies)

    def _phase4_create_unified_namespace(self, full_matches: Dict, unique_concepts: Dict, 
                                       partial_matches: Dict):
        """Phase 4: Create unified namespace for all concepts"""
        
        logger.info("Phase 4: Creating unified namespace...")
        
        concept_counter = 1
        
        # Process full cross-ontology matches
        for concept_label, match_data in full_matches.items():
            unified_id = f"scw_concept_{concept_counter:06d}"
            
            self.unified_concepts[unified_id] = UnifiedConcept(
                concept_id=unified_id,
                type='CROSS_ONTOLOGY',
                original_label=concept_label,
                match_type=match_data['match_type'],
                coverage=match_data['coverage'],
                coverage_percentage=match_data['coverage_percentage'],
                source_entries=match_data['entries'],
                contexts=self._create_context_metadata(match_data['entries']),
                missing_ontologies=None
            )
            concept_counter += 1
        
        # Process partial cross-ontology matches
        for concept_label, match_data in partial_matches.items():
            unified_id = f"scw_concept_{concept_counter:06d}"
            
            self.unified_concepts[unified_id] = UnifiedConcept(
                concept_id=unified_id,
                type='PARTIAL_CROSS_ONTOLOGY',
                original_label=concept_label,
                match_type=match_data['match_type'],
                coverage=match_data['coverage'],
                coverage_percentage=match_data['coverage_percentage'],
                source_entries=match_data['entries'],
                missing_ontologies=match_data['missing_ontologies'], # Store missing ontologies
                contexts=self._create_context_metadata(match_data['entries'])
            )
            concept_counter += 1
        
        # Process unique concepts
        for concept_label, entry in unique_concepts.items():
            unified_id = f"scw_concept_{concept_counter:06d}"
            
            self.unified_concepts[unified_id] = UnifiedConcept(
                concept_id=unified_id,
                type='UNIQUE_CONCEPT',
                original_label=concept_label,
                source_entries=[entry], # Treat as a list for consistency
                match_type='UNIQUE',
                coverage='SINGLE_ONTOLOGY',
                coverage_percentage=(100.0 / len(self.original_ontologies)) if len(self.original_ontologies) > 0 else 0, # Percentage of *loaded* ontologies this unique concept covers
                contexts=self._create_context_metadata([entry]),
                missing_ontologies=[oid for oid in self.original_ontologies.keys() if oid != entry.ontology_id]
            )
            concept_counter += 1
        
        self.processing_stats['total_unified_concepts'] = len(self.unified_concepts)
        
        logger.info(f"Phase 4 complete: Created {len(self.unified_concepts)} unified concepts")

    def _create_context_metadata(self, entries: List[ConceptEntry]) -> Dict[str, Dict]:
        """Create rich context metadata for concept entries"""
        
        contexts = {}
        
        for entry in entries:
            contexts[entry.ontology_id] = {
                'original_uri': entry.concept['uri'],
                'signature': entry.signature,
                'relationship_count': len(entry.relationships),
                'property_count': len(entry.properties), # This is the count of *direct* properties from concept dict
                'properties_summary': {self._extract_local_name(k): v.get('meta', {}).get('type', 'unknown') for k, v in entry.properties.items()},
                'domain_weight': self._calculate_domain_weight(entry),
                'connectivity_score': self._calculate_connectivity_score(entry)
            }
        
        return contexts

    def _calculate_domain_weight(self, entry: ConceptEntry) -> float:
        """Calculate domain weight based on concept centrality (can be enhanced with NetworkX)."""
        
        rel_count = len(entry.relationships)
        prop_count = len(entry.properties)
        
        # Simple heuristic - for more advanced, use networkx on the original graph
        # e.g., degree centrality, pagerank within its original ontology sub-graph
        weight = min(1.0, (rel_count + prop_count) / 15.0)
        return round(weight, 3)

    def _calculate_connectivity_score(self, entry: ConceptEntry) -> float:
        """Calculate how well-connected this concept is (can be enhanced with NetworkX)."""
        
        unique_predicates = len(set(rel['predicate_local'] for rel in entry.relationships))
        total_connections = len(entry.relationships)
        
        if total_connections == 0:
            return 0.0
        
        # Higher score for diverse predicates
        diversity_score = unique_predicates / max(1, total_connections)
        connection_score = min(1.0, total_connections / 10.0)
        
        return round((diversity_score + connection_score) / 2, 3)

    def _phase5_discover_cross_relationships(self):
        """Phase 5: Discover relationships across unified concepts"""
        
        logger.info("Phase 5: Discovering cross-ontology relationships...")
        
        # Create concept URI to unified ID mapping
        uri_to_unified = {}
        for unified_id, concept_data in self.unified_concepts.items():
            for entry in concept_data.source_entries: # Iterate over all source entries
                uri_to_unified[entry.concept['uri']] = unified_id
        
        # Process relationships from all successfully loaded ontologies
        relationship_set = set()  # Avoid duplicates
        
        for ontology_id, ontology_data in self.original_ontologies.items():
            for rel in ontology_data['relationships']:
                subject_unified = uri_to_unified.get(rel['subject'])
                object_unified = uri_to_unified.get(rel['object'])
                
                # Ensure both subject and object map to a unified concept
                if subject_unified and object_unified:
                    # If subject and object map to the same unified concept, it's an internal relationship,
                    # not a 'cross-relationship' in the sense of connecting *different* unified concepts.
                    # We still add it to preserve structure, but mark its origin.
                    
                    # Create relationship tuple for deduplication
                    rel_tuple = (subject_unified, rel['predicate'], object_unified)
                    
                    if rel_tuple not in relationship_set:
                        relationship_set.add(rel_tuple)
                        
                        cross_rel = {
                            'subject': subject_unified,
                            'predicate': rel['predicate'],
                            'predicate_local': rel['predicate_local'],
                            'object': object_unified,
                            'source_ontology': ontology_id,
                            'is_cross_ontology': self._is_cross_ontology_relationship(
                                subject_unified, object_unified, ontology_id
                            ), # Pass source ontology for more accurate check
                            'confidence': 1.0 # All directly extracted relationships have high confidence
                        }
                        
                        self.cross_relationships.append(cross_rel)
        
        logger.info(f"Phase 5 complete: Discovered {len(self.cross_relationships)} relationships")

    def _is_cross_ontology_relationship(self, subject_id: str, object_id: str, source_ont_id: str) -> bool:
        """
        Check if relationship truly crosses ontology boundaries.
        A relationship between A (from OntX) and B (from OntY) is cross-ontology.
        A relationship between A (from OntX, part of a unified concept U) and B (from OntX, part of same unified concept U)
        is not cross-ontology, even if U includes OntZ.
        """
        
        subj_unified_concept = self.unified_concepts[subject_id]
        obj_unified_concept = self.unified_concepts[object_id]

        # Get the original ontology IDs that contribute to the subject and object unified concepts
        subj_source_onts = set(entry.ontology_id for entry in subj_unified_concept.source_entries)
        obj_source_onts = set(entry.ontology_id for entry in obj_unified_concept.source_entries)

        # If the relationship itself came from an ontology that is *not* shared by both unified concepts,
        # or if the unified concepts themselves span different sets of ontologies.
        # This is a bit complex. Simplest: if they are different unified concepts, it's a cross-relationship.
        if subject_id != object_id:
            # If the original source ontology of this relationship is not common to both unified concepts
            # OR if one of the unified concepts does not originate from the source_ont_id
            if source_ont_id not in subj_source_onts or source_ont_id not in obj_source_onts:
                return True
            # If the two unified concepts originate from entirely different sets of ontologies
            if not subj_source_onts.intersection(obj_source_onts):
                return True
        
        return False # If subject == object or they share all source ontologies, it's not "cross"

    def _phase6_create_bridges_and_compensation(self):
        """Phase 6: Create bridges and compensation relationships"""
        
        logger.info("Phase 6: Creating bridges and compensation relationships...")
        
        # Create semantic bridges for unique concepts
        self._create_semantic_bridges()
        
        # Create compensation bridges for partial coverage concepts
        if self.config.get('enable_compensation_bridges', True):
            self._create_compensation_bridges()
        
        total_bridges = len(self.bridge_relationships) + len(self.compensation_bridges)
        logger.info(f"Phase 6 complete: Created {len(self.bridge_relationships)} semantic bridges, "
                   f"{len(self.compensation_bridges)} compensation bridges (Total: {total_bridges})")

    def _create_semantic_bridges(self):
        """Create semantic bridges for unique concepts"""
        
        unique_concepts = [
            (uid, data) for uid, data in self.unified_concepts.items()
            if data.type == 'UNIQUE_CONCEPT'
        ]
        
        bridge_threshold = self.config.get('bridge_threshold', 0.35)
        max_bridges = self.config.get('max_bridges_per_concept', 5)
        
        for unique_id, unique_data in unique_concepts:
            candidates = self._find_bridge_candidates(unique_id, unique_data)
            
            # Filter by threshold and limit
            valid_candidates = [c for c in candidates if c['strength'] >= bridge_threshold]
            limited_candidates = valid_candidates[:max_bridges]
            
            for candidate in limited_candidates:
                bridge = {
                    'subject': unique_id,
                    'predicate': f"{self.scw_namespace}semanticBridge_similarTo", # More specific predicate
                    'object': candidate['target_id'],
                    'bridge_strength': candidate['strength'],
                    'bridge_type': candidate['type'],
                    'justification': candidate['justification'],
                    'discovery_method': 'semantic_similarity'
                }
                
                self.bridge_relationships.append(bridge)

    def _find_bridge_candidates(self, unique_id: str, unique_data: UnifiedConcept) -> List[Dict]:
        """Find bridge candidates for unique concept"""
        
        candidates = []
        unique_entry = unique_data.source_entries[0] # Unique concepts have one source entry
        
        # Prepare comparison sets for the unique concept
        unique_predicates = set(rel['predicate_local'] for rel in unique_entry.relationships)
        unique_properties_fingerprint = set()
        for uri, data in unique_entry.properties.items():
            fingerprint = self._extract_local_name(uri)
            fingerprint += "_" + "_".join(sorted(data.get('meta', {}).get('characteristics', [])))
            unique_properties_fingerprint.add(fingerprint)
        
        unique_label = unique_entry.concept.get('label', unique_entry.concept.get('local_name', ''))
        unique_terms = self._get_synonyms(unique_label)

        rel_weight = self.config.get('similarity_rel_weight', 0.4)
        prop_weight = self.config.get('similarity_prop_weight', 0.25)
        lex_weight = self.config.get('similarity_lex_weight', 0.15) # Exclude signature as it's less direct for cross-type linking

        # Normalize weights for bridge strength
        total_bridge_weight = rel_weight + prop_weight + lex_weight
        if abs(total_bridge_weight - 0.0) < 1e-6: total_bridge_weight = 1.0 # Avoid division by zero
        rel_weight /= total_bridge_weight
        prop_weight /= total_bridge_weight
        lex_weight /= total_bridge_weight
            
        for target_id, target_data in self.unified_concepts.items():
            if target_id == unique_id:
                continue # Don't bridge to itself
            
            # Compare unique concept with other unified concepts
            # For simplicity, compare with first entry of the target unified concept
            # A more thorough comparison would average similarity across all entries in target_data.source_entries
            target_entry_for_comparison = target_data.source_entries[0] 
            
            target_predicates = set(rel['predicate_local'] for rel in target_entry_for_comparison.relationships)
            target_properties_fingerprint = set()
            for uri, data in target_entry_for_comparison.properties.items():
                fingerprint = self._extract_local_name(uri)
                fingerprint += "_" + "_".join(sorted(data.get('meta', {}).get('characteristics', [])))
                target_properties_fingerprint.add(fingerprint)
            
            target_label = target_entry_for_comparison.concept.get('label', target_entry_for_comparison.concept.get('local_name', ''))
            target_terms = self._get_synonyms(target_label)

            pred_sim = self._jaccard_similarity(unique_predicates, target_predicates)
            prop_sim = self._jaccard_similarity(unique_properties_fingerprint, target_properties_fingerprint)
            lexical_sim = self._jaccard_similarity(unique_terms, target_terms)
            
            combined_strength = (pred_sim * rel_weight) + (prop_sim * prop_weight) + (lexical_sim * lex_weight)

            if combined_strength > 0.1:  # Minimum threshold for consideration before sorting
                justification_parts = []
                if pred_sim > 0.0: justification_parts.append(f"Predicates: {round(pred_sim, 2)}")
                if prop_sim > 0.0: justification_parts.append(f"Properties: {round(prop_sim, 2)}")
                if lexical_sim > 0.0: justification_parts.append(f"Lexical: {round(lexical_sim, 2)}")
                
                candidates.append({
                    'target_id': target_id,
                    'strength': combined_strength,
                    'type': 'SEMANTIC_SIMILARITY',
                    'justification': ", ".join(justification_parts) if justification_parts else "Low similarity"
                })
        
        return sorted(candidates, key=lambda x: x['strength'], reverse=True)

    def _create_compensation_bridges(self):
        """Create compensation bridges for partial coverage concepts"""
        
        partial_concepts = [
            (uid, data) for uid, data in self.unified_concepts.items()
            if data.type == 'PARTIAL_CROSS_ONTOLOGY'
        ]
        
        for partial_id, partial_data in partial_concepts:
            missing_ontologies = partial_data.missing_ontologies
            
            for missing_ont_id in missing_ontologies:
                # Find best representative from missing ontology
                representative = self._find_ontology_representative(partial_data, missing_ont_id)
                
                if representative:
                    compensation = {
                        'subject': partial_id,
                        'predicate': f"{self.scw_namespace}compensationBridge_fillsCoverageGap", # More specific predicate
                        'object': representative['concept_id'],
                        'bridge_strength': representative['strength'],
                        'bridge_type': 'COVERAGE_COMPENSATION',
                        'missing_ontology': missing_ont_id,
                        'justification': representative['justification'],
                        'discovery_method': 'coverage_compensation'
                    }
                    
                    self.compensation_bridges.append(compensation)

    def _find_ontology_representative(self, partial_data: UnifiedConcept, missing_ontology_id: str) -> Optional[Dict]:
        """Find best representative concept from a missing ontology for a partial concept."""
        
        # Get sample characteristics from partial concept (using first entry)
        sample_entry = partial_data.source_entries[0]
        sample_predicates = set(rel['predicate_local'] for rel in sample_entry.relationships)
        sample_properties_fingerprint = set()
        for uri, data in sample_entry.properties.items():
            fingerprint = self._extract_local_name(uri)
            fingerprint += "_" + "_".join(sorted(data.get('meta', {}).get('characteristics', [])))
            sample_properties_fingerprint.add(fingerprint)
        
        sample_label = sample_entry.concept.get('label', sample_entry.concept.get('local_name', ''))
        sample_terms = self._get_synonyms(sample_label)

        best_candidate = None
        best_strength = 0.0
        
        # Search through concepts from the *missing* ontology
        for concept_id, concept_data in self.unified_concepts.items():
            # Check if this unified concept has a source entry from the missing ontology
            candidate_entry_from_missing_ont = None
            for entry in concept_data.source_entries:
                if entry.ontology_id == missing_ontology_id:
                    candidate_entry_from_missing_ont = entry
                    break
            
            if candidate_entry_from_missing_ont:
                candidate_predicates = set(rel['predicate_local'] for rel in candidate_entry_from_missing_ont.relationships)
                candidate_properties_fingerprint = set()
                for uri, data in candidate_entry_from_missing_ont.properties.items():
                    fingerprint = self._extract_local_name(uri)
                    fingerprint += "_" + "_".join(sorted(data.get('meta', {}).get('characteristics', [])))
                    candidate_properties_fingerprint.add(fingerprint)
                
                candidate_label = candidate_entry_from_missing_ont.concept.get('label', candidate_entry_from_missing_ont.concept.get('local_name', ''))
                candidate_terms = self._get_synonyms(candidate_label)

                # Use adjusted weights for finding compensation representatives
                strength = (pred_sim * self.config.get('similarity_rel_weight', 0.4)) + \
                           (prop_sim * self.config.get('similarity_prop_weight', 0.25)) + \
                           (lexical_sim * self.config.get('similarity_lex_weight', 0.15))

                if strength > best_strength:
                    best_strength = strength
                    shared_elements_preds = sample_predicates.intersection(candidate_predicates)
                    shared_elements_props = sample_properties_fingerprint.intersection(candidate_properties_fingerprint)
                    shared_elements_lex = sample_terms.intersection(candidate_terms)
                    
                    justification_parts = []
                    if shared_elements_preds: justification_parts.append(f"shared predicates ({len(shared_elements_preds)})")
                    if shared_elements_props: justification_parts.append(f"shared properties ({len(shared_elements_props)})")
                    if shared_elements_lex: justification_parts.append(f"shared lexical terms ({len(shared_elements_lex)})")

                    best_candidate = {
                        'concept_id': concept_id,
                        'strength': strength,
                        'justification': f"Best match in {missing_ontology_id} via: {', '.join(justification_parts) if justification_parts else 'low overlap'}"
                    }
        
        return best_candidate if best_strength > self.config.get('bridge_threshold', 0.35) else None

    def _phase7_generate_unified_owl(self) -> Graph:
        """Phase 7: Generate unified OWL graph"""
        
        logger.info("Phase 7: Generating unified OWL graph...")
        
        # Create new graph
        unified_graph = Graph()
        
        # Bind namespaces
        unified_graph.bind("scw", self.scw_namespace)
        unified_graph.bind("owl", OWL)
        unified_graph.bind("rdf", RDF)
        unified_graph.bind("rdfs", RDFS)
        unified_graph.bind("xsd", XSD) # Bind XSD for literals

        # Add ontology header
        self._add_ontology_header(unified_graph)
        
        # Add unified concepts
        self._add_unified_concepts(unified_graph)
        
        # Add relationships
        self._add_relationships(unified_graph)
        
        # Add bridges
        self._add_bridges(unified_graph)
        
        logger.info(f"Phase 7 complete: Generated OWL with {len(unified_graph)} triples")
        return unified_graph

    def _add_ontology_header(self, graph: Graph):
        """Add ontology declaration and metadata"""
        
        ontology_uri = URIRef(f"{self.scw_namespace}UnifiedOntology")
        
        # Basic ontology declaration
        graph.add((ontology_uri, RDF.type, OWL.Ontology))
        graph.add((ontology_uri, RDFS.label, Literal("SCW Unified Ontology")))
        graph.add((ontology_uri, RDFS.comment, 
                  Literal("Unified ontology created by Semantic Context Weaving algorithm. Designed for ECII processing. Includes robust parsing for input files."))) # Updated comment
        graph.add((ontology_uri, OWL.versionInfo, Literal(f"SCW Algorithm Version {self.config.get('version', '2.1')}"))) # Updated version
        graph.add((ontology_uri, URIRef(f"{self.scw_namespace}generatedOn"), Literal(time.strftime('%Y-%m-%dT%H:%M:%S'), datatype=XSD.dateTime))) # Fix ISO 8601

        # Add processing metadata
        graph.add((ontology_uri, URIRef(f"{self.scw_namespace}totalInputFilesAttempted"),
                  Literal(self.processing_stats.get('total_input_files_attempted', 0), datatype=XSD.integer))) # New stat
        graph.add((ontology_uri, URIRef(f"{self.scw_namespace}successfullyLoadedOntologyCount"), 
                  Literal(len(self.original_ontologies), datatype=XSD.integer)))
        graph.add((ontology_uri, URIRef(f"{self.scw_namespace}unifiedConceptCount"), 
                  Literal(len(self.unified_concepts), datatype=XSD.integer)))
        graph.add((ontology_uri, URIRef(f"{self.scw_namespace}processingScale"), 
                  Literal(self.config.get('scale', 'UNKNOWN'))))
        
        # Add source ontology list
        for ont_id, ont_data in self.original_ontologies.items():
            source_node = BNode()
            graph.add((ontology_uri, URIRef(f"{self.scw_namespace}hasSourceOntology"), source_node))
            graph.add((source_node, RDF.type, self.scw_namespace.SourceOntology))
            graph.add((source_node, self.scw_namespace.ontologyId, Literal(ont_id)))
            graph.add((source_node, self.scw_namespace.filePath, Literal(ont_data['file_path'])))
            graph.add((source_node, self.scw_namespace.conceptCount, 
                      Literal(len(ont_data['concepts']), datatype=XSD.integer)))
            graph.add((source_node, self.scw_namespace.relationshipCount,
                      Literal(len(ont_data['relationships']), datatype=XSD.integer)))
            graph.add((source_node, self.scw_namespace.explicitEquivalencesFound,
                      Literal(len(ont_data['explicit_equivalences']), datatype=XSD.integer)))
        
        # New: Add unparsable files list
        if self.unparsable_files:
            unparsable_list_node = BNode()
            graph.add((ontology_uri, self.scw_namespace.hasUnparsableFiles, unparsable_list_node))
            graph.add((unparsable_list_node, RDF.type, self.scw_namespace.UnparsableFilesList))
            for uf in self.unparsable_files:
                unparsable_item_node = BNode()
                graph.add((unparsable_list_node, self.scw_namespace.unparsableFile, unparsable_item_node))
                graph.add((unparsable_item_node, self.scw_namespace.filePath, Literal(uf['file_path'])))
                graph.add((unparsable_item_node, self.scw_namespace.parsingError, Literal(uf['error'])))


        # Add configuration used
        config_node = BNode()
        graph.add((ontology_uri, self.scw_namespace.processingConfiguration, config_node))
        graph.add((config_node, RDF.type, self.scw_namespace.ProcessingConfiguration))
        for key, value in self.config.items():
            if isinstance(value, (int, float, bool)):
                graph.add((config_node, URIRef(f"{self.scw_namespace}{key}"), Literal(value)))
            else: # Stringify other types
                graph.add((config_node, URIRef(f"{self.scw_namespace}{key}"), Literal(str(value))))

        # Add processing statistics
        stats_node = BNode()
        graph.add((ontology_uri, self.scw_namespace.processingStatistics, stats_node))
        graph.add((stats_node, RDF.type, self.scw_namespace.ProcessingStatistics))
        for key, value in self.processing_stats.items():
            if value is not None:
                if isinstance(value, (int, float)):
                     graph.add((stats_node, URIRef(f"{self.scw_namespace}{key}"), Literal(value)))
                else:
                     graph.add((stats_node, URIRef(f"{self.scw_namespace}{key}"), Literal(str(value))))

    def _add_unified_concepts(self, graph: Graph):
        """Add all unified concepts to graph"""
        
        for concept_id, concept_data in self.unified_concepts.items():
            concept_uri = URIRef(f"{self.scw_namespace}{concept_id}")
            
            # Basic concept declaration
            graph.add((concept_uri, RDF.type, OWL.Class))
            graph.add((concept_uri, RDFS.label, Literal(concept_data.original_label)))
            
            # Add SCW metadata
            graph.add((concept_uri, self.scw_namespace.conceptType, Literal(concept_data.type)))
            graph.add((concept_uri, self.scw_namespace.matchType, Literal(concept_data.match_type)))
            graph.add((concept_uri, self.scw_namespace.coverage, Literal(concept_data.coverage)))
            graph.add((concept_uri, self.scw_namespace.coveragePercentage, 
                      Literal(concept_data.coverage_percentage, datatype=XSD.float)))
            
            # Add context information for each source ontology
            for ont_id, context in concept_data.contexts.items():
                context_node = BNode()
                graph.add((concept_uri, self.scw_namespace.hasContext, context_node))
                graph.add((context_node, RDF.type, self.scw_namespace.ConceptContext))
                graph.add((context_node, self.scw_namespace.sourceOntology, Literal(ont_id)))
                graph.add((context_node, self.scw_namespace.originalURI, URIRef(context['original_uri'])))
                graph.add((context_node, self.scw_namespace.signature, Literal(context['signature']))) # Add signature
                graph.add((context_node, self.scw_namespace.relationshipCount, 
                          Literal(context['relationship_count'], datatype=XSD.integer)))
                graph.add((context_node, self.scw_namespace.propertyCount, 
                          Literal(context['property_count'], datatype=XSD.integer)))
                graph.add((context_node, self.scw_namespace.domainWeight, 
                          Literal(context['domain_weight'], datatype=XSD.float)))
                graph.add((context_node, self.scw_namespace.connectivityScore, 
                          Literal(context['connectivity_score'], datatype=XSD.float)))
                
                # Add properties summary
                if context['properties_summary']:
                    props_summary_node = BNode()
                    graph.add((context_node, self.scw_namespace.propertiesSummary, props_summary_node))
                    for prop_name, prop_type in context['properties_summary'].items():
                        graph.add((props_summary_node, URIRef(self.scw_namespace + prop_name), Literal(prop_type)))
            
            # Add missing ontologies for partial and unique concepts
            if concept_data.missing_ontologies:
                for missing_ont in concept_data.missing_ontologies:
                    graph.add((concept_uri, self.scw_namespace.missingFromOntology, Literal(missing_ont)))
            
            # For Unique Concepts, add specific metadata (redundant if already covered by context but useful for clarity)
            if concept_data.type == 'UNIQUE_CONCEPT':
                graph.add((concept_uri, self.scw_namespace.sourceOntology, Literal(concept_data.source_entries[0].ontology_id)))
                graph.add((concept_uri, self.scw_namespace.preservationStatus, Literal("DOMAIN_EXCLUSIVE")))
                graph.add((concept_uri, self.scw_namespace.originalURI, URIRef(concept_data.source_entries[0].concept['uri'])))


    def _add_relationships(self, graph: Graph):
        """Add cross-ontology relationships to graph"""
        
        for rel in self.cross_relationships:
            subject_uri = URIRef(f"{self.scw_namespace}{rel['subject']}")
            predicate_uri = URIRef(rel['predicate']) # Use original predicate URI
            object_uri = URIRef(f"{self.scw_namespace}{rel['object']}")
            
            # Add the relationship
            graph.add((subject_uri, predicate_uri, object_uri))
            
            # Add relationship metadata if cross-ontology or for tracking
            rel_meta_node = BNode()
            graph.add((subject_uri, self.scw_namespace.hasDiscoveredRelationship, rel_meta_node))
            graph.add((rel_meta_node, RDF.type, self.scw_namespace.DiscoveredRelationship))
            graph.add((rel_meta_node, self.scw_namespace.relPredicate, predicate_uri))
            graph.add((rel_meta_node, self.scw_namespace.relObject, object_uri))
            graph.add((rel_meta_node, self.scw_namespace.sourceOntology, 
                          Literal(rel['source_ontology'])))
            graph.add((rel_meta_node, self.scw_namespace.isCrossOntology, Literal(rel['is_cross_ontology'], datatype=XSD.boolean)))
            graph.add((rel_meta_node, self.scw_namespace.confidence, Literal(rel['confidence'], datatype=XSD.float)))


    def _add_bridges(self, graph: Graph):
        """Add bridge relationships to graph"""
        
        # Add semantic bridges
        for bridge in self.bridge_relationships:
            subject_uri = URIRef(f"{self.scw_namespace}{bridge['subject']}")
            predicate_uri = URIRef(bridge['predicate']) # Will be scw:semanticBridge_similarTo
            object_uri = URIRef(f"{self.scw_namespace}{bridge['object']}")
            
            graph.add((subject_uri, predicate_uri, object_uri)) # Add the direct bridge triple
            
            # Add bridge metadata
            bridge_node = BNode()
            graph.add((subject_uri, self.scw_namespace.hasSemanticBridge, bridge_node))
            graph.add((bridge_node, RDF.type, self.scw_namespace.SemanticBridge))
            graph.add((bridge_node, self.scw_namespace.bridgeToConcept, object_uri))
            graph.add((bridge_node, self.scw_namespace.bridgeStrength, 
                      Literal(bridge['bridge_strength'], datatype=XSD.float)))
            graph.add((bridge_node, self.scw_namespace.bridgeType, 
                      Literal(bridge['bridge_type'])))
            graph.add((bridge_node, self.scw_namespace.justification, 
                      Literal(bridge['justification'])))
            graph.add((bridge_node, self.scw_namespace.discoveryMethod, 
                      Literal(bridge['discovery_method'])))
        
        # Add compensation bridges
        for comp in self.compensation_bridges:
            subject_uri = URIRef(f"{self.scw_namespace}{comp['subject']}")
            predicate_uri = URIRef(comp['predicate']) # Will be scw:compensationBridge_fillsCoverageGap
            object_uri = URIRef(f"{self.scw_namespace}{comp['object']}")
            
            graph.add((subject_uri, predicate_uri, object_uri)) # Add the direct compensation triple
            
            # Add compensation metadata
            comp_node = BNode()
            graph.add((subject_uri, self.scw_namespace.hasCompensationBridge, comp_node))
            graph.add((comp_node, RDF.type, self.scw_namespace.CompensationBridge))
            graph.add((comp_node, self.scw_namespace.bridgeToConcept, object_uri))
            graph.add((comp_node, self.scw_namespace.bridgeStrength, 
                      Literal(comp['bridge_strength'], datatype=XSD.float)))
            graph.add((comp_node, self.scw_namespace.bridgeType, 
                      Literal(comp['bridge_type'])))
            graph.add((comp_node, self.scw_namespace.missingOntology, 
                      Literal(comp['missing_ontology'])))
            graph.add((comp_node, self.scw_namespace.justification, 
                      Literal(comp['justification'])))

    def _phase8_save_and_validate(self, unified_graph: Graph, output_file: str) -> Dict[str, Any]:
        """Phase 8: Save unified OWL and perform validation"""
        
        logger.info("Phase 8: Saving and validating unified ontology...")
        
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save unified OWL
            unified_graph.serialize(destination=output_file, format='json-ld')
            
            # Perform validation
            validation_report = self._validate_unified_ontology(unified_graph, output_file) # Pass output_file for reasoner
            
            # Save validation report
            report_file = output_path.with_suffix('.validation.json')
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to: {report_file}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Failed to save or validate: {str(e)}")
            raise

    def _validate_unified_ontology(self, unified_graph: Graph, output_file_path: str) -> Dict[str, Any]:
        """
        Comprehensive validation of unified ontology, including optional reasoning via subprocess call.
        """
        
        # Count elements
        total_concepts = len(list(unified_graph.subjects(RDF.type, OWL.Class)))
        total_triples = len(unified_graph)
        
        # Calculate preservation metrics
        original_concept_count = self.processing_stats['total_input_concepts']
        original_relationship_count = self.processing_stats['total_input_relationships']
        
        concept_preservation_rate = (total_concepts / original_concept_count) if original_concept_count > 0 else 0
        
        # Calculate enhancement metrics
        cross_relationships = len(self.cross_relationships)
        bridge_relationships = len(self.bridge_relationships)
        compensation_bridges = len(self.compensation_bridges)
        total_enhancements = cross_relationships + bridge_relationships + compensation_bridges
        
        # Validate OWL syntax (basic rdflib check implicitly done during parsing/serialization)
        owl_valid = True
        owl_errors = []
        try:
            ontology_count = len(list(unified_graph.subjects(RDF.type, OWL.Ontology)))
            if ontology_count != 1:
                owl_errors.append(f"Expected 1 ontology declaration, found {ontology_count}")
                owl_valid = False
            
        except Exception as e:
            owl_errors.append(f"Basic OWL validation error: {str(e)}")
            owl_valid = False
        
        # --- NEW: Reasoner Integration via Subprocess Call (Bypassing Owlready2 API) ---
        if self.config.get('enable_reasoning_validation', False):
            logger.info("Running OWL reasoning for enhanced validation via external HermiT process...")
            try:
                script_dir = Path(__file__).parent
                hermit_dir = script_dir / "reasoners"
                
                if not hermit_dir.exists():
                    raise FileNotFoundError(f"Reasoner directory not found: {hermit_dir}. Please create it and place HermiT.jar inside.")
                
                hermit_jar_files = list(hermit_dir.glob("HermiT*.jar")) + list(hermit_dir.glob("hermit*.jar"))
                
                if not hermit_jar_files:
                    raise FileNotFoundError(f"No HermiT.jar file found in '{hermit_dir}'. Please download and place HermiT.jar there.")
                
                hermit_jar_path = str(hermit_jar_files[0].resolve())
                
                # Check if Java is in PATH
                try:
                    subprocess.run(['java', '-version'], capture_output=True, check=True, text=True)
                except FileNotFoundError:
                    raise FileNotFoundError("Java executable ('java.exe' or 'java') not found in system PATH. Please install Java and add it to PATH.")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Java command failed: {e.stderr.strip()}. Check Java installation.")

                logger.info(f"Invoking HermiT: java -jar {hermit_jar_path} {output_file_path}")
                
                # Run HermiT as a subprocess
                # Using Popen for more control and consistent STDOUT/STDERR handling across Python versions
                # HermiT might output to stderr, so redirect stderr to stdout to capture all messages.
                process = Popen(
                    ['java', '-jar', hermit_jar_path, output_file_path],
                    stdout=PIPE,
                    stderr=STDOUT, # Redirect stderr to stdout to capture all output in one stream
                    text=True      # Decode output as text
                )
                stdout_data, _ = process.communicate() # Get combined stdout/stderr
                
                
                # Create a fixed-name output file for HermiT's results (will overwrite on each run)
                hermit_output_file = Path(output_file_path).parent / "hermit_reasoning_output.txt"
                with open(hermit_output_file, "w") as f_out:
                    f_out.write(stdout_data)
                
                # Analyze HermiT's output for known inconsistency messages
                # HermiT typically prints "inconsistent" or "unsatisfiable" if issues are found.
                if process.returncode != 0:
                    owl_errors.append(f"HermiT reasoner exited with non-zero code ({process.returncode}). Check {hermit_output_file} for full details.")
                    owl_valid = False
                
                if "unsatisfiable" in stdout_data.lower() or "inconsistent" in stdout_data.lower():
                    owl_errors.append(f"HermiT reasoner reported inconsistencies. Check {hermit_output_file} for full output.")
                    owl_valid = False
                
                logger.info(f"HermiT reasoner execution complete. Output logged to {hermit_output_file}")

            except FileNotFoundError as e:
                owl_errors.append(f"Reasoner setup failed (File Not Found): {str(e)}. Please ensure HermiT.jar is in '{hermit_dir}' and Java is installed/in PATH.")
                logger.error(f"Reasoner setup failed: {str(e)}", exc_info=True)
            except RuntimeError as e: # Catch the custom RuntimeError for Java command failed
                owl_errors.append(f"Java execution error: {str(e)}. Check Java installation and PATH.")
                logger.error(f"Java execution error: {str(e)}", exc_info=True)
            except Exception as e:
                owl_errors.append(f"Reasoner integration failed due to an unexpected error: {str(e)}. Check HermiT setup.")
                logger.error(f"Reasoner integration failed: {str(e)}", exc_info=True)
        else: # Case where reasoning is disabled or Owlready2 is not available and config does not enable subprocess
            if self.config.get('enable_reasoning_validation', False): # Only warn if user *tried* to enable it
                owl_errors.append("Reasoner validation skipped: Not enabled in configuration or required tools (Java/HermiT) are missing/misconfigured. Check logs.")
                logger.warning("Reasoner validation skipped: Not enabled in configuration or required tools (Java/HermiT) are missing/misconfigured. Check logs.")


        validation_report = {
            'validation_timestamp': time.time(),
            'input_statistics': {
                'total_input_files_attempted': self.processing_stats['total_input_files_attempted'],
                'successfully_loaded_ontologies': len(self.original_ontologies),
                'unparsable_input_files': self.unparsable_files,
                'original_concepts_from_loaded_files': original_concept_count,
                'original_relationships_from_loaded_files': original_relationship_count,
                'explicit_equivalences_found_in_loaded_files': self.processing_stats['explicit_equivalences_found']
            },
            'output_statistics': {
                'unified_concepts': total_concepts,
                'total_triples': total_triples,
                'cross_relationships': cross_relationships,
                'bridge_relationships': bridge_relationships,
                'compensation_bridges': compensation_bridges,
                'total_enhancements': total_enhancements
            },
            'preservation_metrics': {
                'concept_preservation_rate': round(concept_preservation_rate, 4),
                'concepts_preserved_at_least_one': concept_preservation_rate >= 1.0,
                'enhancement_ratio_per_concept': round(total_enhancements / total_concepts, 4) if total_concepts > 0 else 0
            },
            'quality_metrics': {
                'owl_syntax_valid': owl_valid,
                'owl_errors_and_warnings': owl_errors,
                'concept_distribution': self._analyze_concept_distribution(),
                'relationship_distribution': self._analyze_relationship_distribution()
            },
            'completeness_verification': {
                'all_concepts_unified_or_preserved': total_concepts > 0,
                'cross_ontology_coverage_ratio': self._calculate_cross_ontology_coverage(),
                'unique_concept_bridge_connectivity_ratio': self._calculate_bridge_connectivity()
            }
        }
        
        return validation_report

    def _analyze_concept_distribution(self) -> Dict[str, int]:
        """Analyze distribution of concept types in the unified ontology."""
        
        distribution = Counter()
        for concept_data in self.unified_concepts.values():
            distribution[concept_data.type] += 1
        
        return dict(distribution)

    def _analyze_relationship_distribution(self) -> Dict[str, int]:
        """Analyze distribution of relationship types in the unified ontology."""
        
        distribution = {
            'original_relationships_retained': len(self.cross_relationships), # This includes both 'cross' and 'same' types that were re-mapped
            'explicit_cross_ontology_relationships': len([r for r in self.cross_relationships if r['is_cross_ontology']]),
            'semantic_bridges_created': len(self.bridge_relationships),
            'compensation_bridges_created': len(self.compensation_bridges)
        }
        
        return distribution

    def _calculate_cross_ontology_coverage(self) -> float:
        """Calculate how well concepts are covered across original ontologies."""
        
        total_unified_concepts = len(self.unified_concepts)
        cross_ontology_concepts = len([
            c for c in self.unified_concepts.values()
            if c.type in ['CROSS_ONTOLOGY', 'PARTIAL_CROSS_ONTOLOGY']
        ])
        
        return round(cross_ontology_concepts / total_unified_concepts, 4) if total_unified_concepts > 0 else 0.0

    def _calculate_bridge_connectivity(self) -> float:
        """Calculate how many unique concepts are connected via bridges."""
        
        unique_concepts_uids = {uid for uid, c_data in self.unified_concepts.items() if c_data.type == 'UNIQUE_CONCEPT'}
        if not unique_concepts_uids:
            return 1.0 # All (zero) unique concepts are considered connected
        
        bridged_unique_concepts_uids = set()
        for bridge in self.bridge_relationships + self.compensation_bridges:
            if bridge['subject'] in unique_concepts_uids:
                bridged_unique_concepts_uids.add(bridge['subject'])
        
        connectivity = len(bridged_unique_concepts_uids) / len(unique_concepts_uids)
        return round(connectivity, 4)

    def _finalize_stats(self) -> Dict[str, Any]:
        """Finalize processing statistics"""
        
        end_time = self.processing_stats['end_time']
        start_time = self.processing_stats['start_time']
        
        processing_time = round(end_time - start_time, 2) if end_time and start_time else 0.0

        self.processing_stats.update({
            'processing_time_seconds': processing_time,
            'concepts_per_second': round(self.processing_stats['total_input_concepts'] / processing_time, 2) if processing_time > 0 else 0,
            'ontologies_processed': len(self.original_ontologies) # This is count of successfully loaded
        })
        
        return self.processing_stats


# Main execution functions
def run_scw_algorithm(ontology_files: List[str], output_file: str = "unified_ontology.owl", 
                     config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main function to run SCW algorithm - HANDLES ANY NUMBER OF ONTOLOGIES
    
    Args:
        ontology_files: List of paths to OWL files (2 to unlimited)
        output_file: Path for output unified OWL file
        config: Optional configuration parameters
    
    Returns:
        Dictionary with execution results
    """
    
    # Basic check for enough file paths provided initially
    if len(ontology_files) < 2:
        return {
            'status': 'error',
            'error': 'At least 2 ontology file paths must be provided to the algorithm for a meaningful attempt at unification.'
        }
    
    logger.info(f"SCW Algorithm starting with {len(ontology_files)} input files...")
    
    processor = SCWProcessor(config)
    return processor.execute(ontology_files, output_file, config)


def discover_and_process_ontologies(directory_path: str, output_file: str = "unified_discovered.owl", 
                                   file_patterns: List[str] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Discover OWL files in directory and process them
    
    Args:
        directory_path: Directory to search for OWL files
        output_file: Output file path
        file_patterns: List of file patterns to match (e.g., ['*.owl', '*.ttl']). Default: common RDF formats.
        config: Optional configuration
    
    Returns:
        Processing results
    """
    
    search_path = Path(directory_path)
    if not file_patterns:
        file_patterns = ["*.owl", "*.ttl", "*.rdf", "*.xml", "*.jsonld"] # Common RDF formats

    ontology_files = []
    for pattern in file_patterns:
        ontology_files.extend(list(search_path.rglob(pattern)))

    ontology_paths = [str(f) for f in ontology_files]
    
    logger.info(f"Discovered {len(ontology_paths)} ontology files in {directory_path} using patterns {file_patterns}")
    
    # This initial check is against discovered files. The actual validation for 
    # minimum required successful parses happens inside processor.execute()
    if len(ontology_paths) < 2:
        return {
            'status': 'error',
            'error': f'Found only {len(ontology_paths)} potentially parsable files in "{directory_path}", need at least 2 for unification.'
        }
    
    return run_scw_algorithm(ontology_paths, output_file, config)


def batch_process_ontology_sets(ontology_sets: List[Dict], base_output_dir: str = "scw_outputs") -> List[Dict]:
    """
    Process multiple sets of ontologies in batch
    
    Args:
        ontology_sets: List of dicts with 'name', 'files', and optional 'config'
        base_output_dir: Base directory for outputs
    
    Returns:
        List of processing results
    """
    
    results = []
    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, ont_set in enumerate(ontology_sets):
        set_name = ont_set.get('name', f'set_{i+1}')
        files = ont_set['files']
        config = ont_set.get('config')
        
        output_file = output_dir / f"{set_name}_unified.owl"
        
        logger.info(f"Processing ontology set: {set_name}")
        result = run_scw_algorithm(files, str(output_file), config)
        result['set_name'] = set_name
        results.append(result)
    
    return results


# Additional utility functions for real-world usage
def validate_ontology_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    Perform a pre-processing validation check on ontology files.
    This check does NOT stop the main algorithm; it just provides a report.
    The main algorithm has its own robust parsing logic.
    
    Args:
        file_paths: List of ontology file paths
    
    Returns:
        Validation results (does not raise errors, just reports)
    """
    
    validation_results = {
        'valid_files': [],
        'invalid_syntax_files': [], # Changed from 'invalid_files'
        'missing_files': [],
        'warnings': [],
        'total_files_checked': len(file_paths),
        'total_concepts_found': 0,
        'total_triples_found': 0
    }
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                validation_results['missing_files'].append(file_path)
                continue
            
            # Try to parse the file
            graph = Graph()
            try:
                graph.parse(file_path)
            except ParserError as e:
                validation_results['invalid_syntax_files'].append({
                    'file_path': file_path,
                    'error': str(e)
                })
                continue # Skip to next file if syntax is invalid
            except Exception as e:
                validation_results['invalid_syntax_files'].append({
                    'file_path': file_path,
                    'error': f"Unexpected parsing error: {str(e)}"
                })
                continue
            
            # Count concepts and triples if parsing was successful
            concepts = len(list(graph.subjects(RDF.type, OWL.Class)))
            triples = len(graph)
            
            validation_results['valid_files'].append({
                'file_path': file_path,
                'concepts': concepts,
                'triples': triples
            })
            
            validation_results['total_concepts_found'] += concepts
            validation_results['total_triples_found'] += triples
            
            # Check for warnings (e.g., empty OWL files)
            if concepts == 0 and triples == 0:
                validation_results['warnings'].append(f"No OWL classes or RDF triples found in {file_path}. File might be empty or malformed.")
            elif concepts == 0:
                validation_results['warnings'].append(f"No OWL classes found in {file_path}, but triples exist. Might be an RDF graph, not a rich OWL ontology.")
            
        except Exception as e:
            # Catch any unexpected errors during this pre-validation process
            logger.error(f"Error during pre-validation of {file_path}: {str(e)}", exc_info=True)
            validation_results['invalid_syntax_files'].append({
                'file_path': file_path,
                'error': f"Unhandled validation error: {str(e)}"
            })
    
    # Determine overall 'validity' for reporting purposes in CLI
    validation_results['is_valid'] = (
        len(validation_results['valid_files']) >= 2 and
        len(validation_results['invalid_syntax_files']) == 0 and
        len(validation_results['missing_files']) == 0
    )
    
    return validation_results



# Configuration presets for different use cases
CONFIGURATION_PRESETS = {
    'research_lab': {
        'similarity_threshold': 0.3,
        'bridge_threshold': 0.25,
        'max_bridges_per_concept': 8,
        'enable_compensation_bridges': True,
        'processing_mode': 'memory',
        'similarity_rel_weight': 0.5, 'similarity_prop_weight': 0.25,
        'similarity_sig_weight': 0.15, 'similarity_lex_weight': 0.1,
        'enable_reasoning_validation': False
    },
    
    'enterprise_integration': {
        'similarity_threshold': 0.4,
        'bridge_threshold': 0.35,
        'max_bridges_per_concept': 6,
        'enable_compensation_bridges': True,
        'processing_mode': 'memory', # Or 'streaming' conceptually
        'similarity_rel_weight': 0.4, 'similarity_prop_weight': 0.2,
        'similarity_sig_weight': 0.25, 'similarity_lex_weight': 0.15,
        'enable_reasoning_validation': True # Good for enterprise to ensure consistency
    },
    
    'knowledge_graph': {
        'similarity_threshold': 0.5,
        'bridge_threshold': 0.4,
        'max_bridges_per_concept': 10,
        'enable_compensation_bridges': True,
        'processing_mode': 'memory', # Or 'batch' conceptually
        'similarity_rel_weight': 0.35, 'similarity_prop_weight': 0.15,
        'similarity_sig_weight': 0.3, 'similarity_lex_weight': 0.2,
        'enable_reasoning_validation': True
    },
    
    'fast_processing': {
        'similarity_threshold': 0.6,
        'bridge_threshold': 0.5,
        'max_bridges_per_concept': 4,
        'enable_compensation_bridges': False,
        'processing_mode': 'memory',
        'similarity_rel_weight': 0.6, 'similarity_prop_weight': 0.2,
        'similarity_sig_weight': 0.1, 'similarity_lex_weight': 0.1,
        'enable_reasoning_validation': False
    },
    
    'comprehensive': {
        'similarity_threshold': 0.2,
        'bridge_threshold': 0.15,
        'max_bridges_per_concept': 15,
        'enable_compensation_bridges': True,
        'processing_mode': 'memory', # Or 'batch' conceptually
        'similarity_rel_weight': 0.3, 'similarity_prop_weight': 0.2,
        'similarity_sig_weight': 0.25, 'similarity_lex_weight': 0.25,
        'enable_reasoning_validation': True
    }
}


def run_scw_with_preset(ontology_files: List[str], preset_name: str, 
                       output_file: str = "unified_ontology.owl") -> Dict[str, Any]:
    """
    Run SCW algorithm with predefined configuration preset
    
    Args:
        ontology_files: List of ontology file paths
        preset_name: Configuration preset name
        output_file: Output file path
    
    Returns:
        Processing results
    """
    
    if preset_name not in CONFIGURATION_PRESETS:
        available_presets = list(CONFIGURATION_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    config = CONFIGURATION_PRESETS[preset_name].copy()
    config['preset_name'] = preset_name
    
    logger.info(f"Running SCW with preset: {preset_name}")
    return run_scw_algorithm(ontology_files, output_file, config)


# Command-line interface support
def main():
    """Command-line interface for SCW algorithm"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic Context Weaving Algorithm')
    
    # Mutually exclusive group for files or discover
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', nargs='*', help='Space-separated list of ontology files to process')
    group.add_argument('-d', '--discover', help='Discover ontologies in specified directory')

    parser.add_argument('-o', '--output', default='unified_ontology.owl', 
                       help='Output file path for the unified OWL ontology')
    parser.add_argument('-p', '--preset', choices=list(CONFIGURATION_PRESETS.keys()),
                       help='Configuration preset to use (e.g., enterprise_integration)')
    
    parser.add_argument('-v', '--validate-inputs', action='store_true',
                       help='Perform a preliminary validation check on input files before processing')
    parser.add_argument('-r', '--report', help='Generate a comprehensive HTML report file at the specified path')
    
    # Overriding individual config parameters
    parser.add_argument('--similarity-threshold', type=float,
                       help='Override the similarity threshold (0.0-1.0) for concept matching')
    parser.add_argument('--bridge-threshold', type=float,
                       help='Override the bridge creation threshold (0.0-1.0) for semantic bridges')
    parser.add_argument('--max-bridges', type=int, dest='max_bridges_per_concept',
                       help='Override the maximum number of bridges to create per unique concept')
    parser.add_argument('--enable-compensation', type=lambda x: (str(x).lower() == 'true'), default=None,
                       help='Enable or disable compensation bridges (true/false)')
    parser.add_argument('--enable-reasoning', type=lambda x: (str(x).lower() == 'true'), default=None,
                       help='Enable OWL reasoning for final validation (true/false, requires owlready2)')
    
    args = parser.parse_args()
    
    # Prepare configuration (start with preset if any, then CLI overrides)
    config_from_cli = {}
    if args.preset:
        config_from_cli.update(CONFIGURATION_PRESETS[args.preset].copy())
    
    if args.similarity_threshold is not None:
        config_from_cli['similarity_threshold'] = args.similarity_threshold
    if args.bridge_threshold is not None:
        config_from_cli['bridge_threshold'] = args.bridge_threshold
    if args.max_bridges_per_concept is not None:
        config_from_cli['max_bridges_per_concept'] = args.max_bridges_per_concept
    if args.enable_compensation is not None:
        config_from_cli['enable_compensation_bridges'] = args.enable_compensation
    if args.enable_reasoning is not None:
        config_from_cli['enable_reasoning_validation'] = args.enable_reasoning
    
    result = None

    # Discover files if requested
    if args.discover:
        result = discover_and_process_ontologies(args.discover, args.output, config=config_from_cli)
    else:
        # Validate files if requested (pre-processing validation)
        if args.validate_inputs:
            print("\n--- Performing preliminary input file validation ---")
            validation_report_cli = validate_ontology_files(args.files)
            if not validation_report_cli['is_valid']:
                print("Input file validation failed. Issues found:")
                for error_item in validation_report_cli['invalid_syntax_files']:
                    print(f"  - Invalid syntax: {error_item.get('file_path')} - {error_item.get('error')}")
                for missing_file in validation_report_cli['missing_files']:
                    print(f"  - Missing: {missing_file}")
                for warning in validation_report_cli['warnings']:
                    print(f"  - Warning: {warning}")
                print("\nSCW algorithm will attempt to process valid files, but consider fixing the issues for best results.")
            else:
                print("Input files validated successfully.")
            print("----------------------------------------------------\n")
        
        # Run main processing
        result = run_scw_algorithm(args.files, args.output, config=config_from_cli)
    
    # Print results and generate report
    if result and result['status'] == 'success':
        print(f"\n Success! Unified ontology created: {result['output_file']}")
        
        # Check if any files were unparsable
        if result.get('unparsable_files'):
            print(f"Warning: {len(result['unparsable_files'])} input file(s) could not be parsed and were skipped.")
            for uf in result['unparsable_files']:
                print(f"  - Skipped: {uf['file_path']} (Error: {uf['error']})")
        
        
        
        # Print key stats from report
        final_stats = result.get('processing_stats', {})
        val_report = result.get('validation_report', {})
        print("\n--- Key Processing Statistics ---")
        print(f"  Total Input Files Attempted: {val_report.get('input_statistics', {}).get('total_input_files_attempted')}")
        print(f"  Successfully Loaded Ontologies: {val_report.get('input_statistics', {}).get('successfully_loaded_ontologies')}")
        print(f"  Total Concepts (from loaded files): {val_report.get('input_statistics', {}).get('original_concepts_from_loaded_files')}")
        print(f"  Total Unified Concepts (Output): {final_stats.get('total_unified_concepts')}")
        print(f"  Processing Time: {final_stats.get('processing_time_seconds')} seconds")
        print(f"  Concept Preservation Rate: {val_report.get('preservation_metrics', {}).get('concept_preservation_rate')}")
        print(f"  Bridge Connectivity Ratio: {val_report.get('completeness_verification', {}).get('unique_concept_bridge_connectivity_ratio')}")
        
        if val_report.get('quality_metrics', {}).get('owl_errors_and_warnings'):
            print(f" Output OWL has validation issues/warnings. Check report for details.")

        return 0
    elif result: # Error occurred during processing
        print(f"\nError: {result['error']}")
        if result.get('unparsable_files'):
             print(f"  {len(result['unparsable_files'])} input file(s) could not be parsed:")
             for uf in result['unparsable_files']:
                 print(f"  - {uf['file_path']} (Error: {uf['error']})")
        print("\nProcessing terminated. Please review the error message.")
        return 1
    else: # Should not happen if result is always returned
        print("\nAn unexpected error occurred and no result was returned.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())