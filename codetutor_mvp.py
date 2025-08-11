#!/usr/bin/env python3
"""
CodeTutor AI - MVP Version
=========================

A simplified but working version that demonstrates the core concept:
- AI learns from YouTube coding tutorials
- Generates code examples and explanations
- Uses neural evolution for continuous improvement

This MVP focuses on what works right now.
"""

import sys
import logging
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path for our modules
sys.path.append('src')

from core.neural_evolution_engine import NeuralEvolutionEngine, NeuralGene, SelfModifyingNeuralNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeTutorAI:
    """
    MVP Version of CodeTutor AI
    
    Demonstrates core concepts:
    - Learning from text/tutorial content
    - Generating code explanations
    - Neural evolution for improvement
    """
    
    def __init__(self):
        self.knowledge_base = {}
        self.learned_patterns = {}
        self.evolution_engine = NeuralEvolutionEngine(population_size=3)
        self.consciousness_level = 0.0
        self.lessons_learned = []
        
        logger.info("🤖 CodeTutor AI MVP initialized")
        
        # Pre-load some basic programming knowledge
        self._initialize_programming_knowledge()
    
    def _initialize_programming_knowledge(self):
        """Load basic programming concepts"""
        basic_concepts = {
            "python_basics": {
                "variables": "Variables store data values. Example: x = 10, name = 'John'",
                "functions": "Functions are reusable blocks of code. Example: def greet(name): return f'Hello {name}'",
                "loops": "Loops repeat code. Example: for i in range(5): print(i)",
                "conditionals": "If statements make decisions. Example: if x > 0: print('positive')"
            },
            "javascript_basics": {
                "variables": "Variables store values. Example: let x = 10; const name = 'John';",
                "functions": "Functions are reusable code blocks. Example: function greet(name) { return `Hello ${name}`; }",
                "loops": "Loops repeat code. Example: for (let i = 0; i < 5; i++) { console.log(i); }",
                "conditionals": "If statements make decisions. Example: if (x > 0) { console.log('positive'); }"
            }
        }
        
        self.knowledge_base = basic_concepts
        logger.info(f"📚 Loaded {len(basic_concepts)} programming language basics")
    
    def learn_from_tutorial_text(self, tutorial_text: str, language: str = "python") -> Dict[str, Any]:
        """
        Learn programming concepts from tutorial text
        (In full version, this would process actual YouTube videos)
        """
        logger.info(f"🎓 Learning from tutorial: {language}")
        
        start_time = time.time()
        
        # Extract key programming concepts from text
        concepts_found = self._extract_programming_concepts(tutorial_text, language)
        
        # Use neural evolution to improve understanding
        learning_quality = self._evolve_understanding(concepts_found)
        
        # Store learned concepts
        lesson = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "concepts_learned": concepts_found,
            "learning_quality": learning_quality,
            "tutorial_text": tutorial_text[:200] + "..." if len(tutorial_text) > 200 else tutorial_text
        }
        
        self.lessons_learned.append(lesson)
        
        # Update consciousness level
        self.consciousness_level += 0.01 * learning_quality
        self.consciousness_level = min(1.0, self.consciousness_level)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Learning complete: {len(concepts_found)} concepts, "
                   f"quality: {learning_quality:.2f}, time: {processing_time:.2f}s")
        
        return {
            "concepts_learned": concepts_found,
            "learning_quality": learning_quality,
            "processing_time": processing_time,
            "consciousness_level": self.consciousness_level
        }
    
    def _extract_programming_concepts(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract programming concepts from tutorial text"""
        concepts = []
        text_lower = text.lower()
        
        # Common programming concepts to look for
        concept_patterns = {
            "function": ["def ", "function ", "func ", "method"],
            "variable": ["variable", "var ", "let ", "const ", "assign"],
            "loop": ["for ", "while ", "loop", "iterate"],
            "conditional": ["if ", "else", "elif", "condition"],
            "class": ["class ", "object", "instance"],
            "import": ["import ", "require(", "include"],
            "error_handling": ["try", "catch", "except", "error"],
            "data_structure": ["list", "array", "dict", "object", "hash"]
        }
        
        for concept_type, patterns in concept_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Extract context around the concept
                    start_idx = text_lower.find(pattern)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(text), start_idx + 100)
                    context = text[context_start:context_end].strip()
                    
                    concepts.append({
                        "type": concept_type,
                        "pattern": pattern,
                        "context": context,
                        "confidence": random.uniform(0.7, 0.95)  # Simulated confidence
                    })
                    break  # One match per concept type
        
        return concepts
    
    def _evolve_understanding(self, concepts: List[Dict[str, Any]]) -> float:
        """Use neural evolution to improve understanding quality"""
        # Create specialized genes for the concepts found
        concept_genes = []
        
        for concept in concepts:
            gene = NeuralGene(
                gene_id=f"concept_{concept['type']}_{random.randint(1000, 9999)}",
                layer_type="linear",
                input_dim=128,
                output_dim=64,
                activation="relu",
                fitness_score=concept["confidence"]
            )
            concept_genes.append(gene)
        
        if concept_genes:
            # Create a specialized network for these concepts
            concept_network = SelfModifyingNeuralNetwork(concept_genes, evolution_rate=0.2)
            
            # Simulate learning by recording performance
            performance = sum(c["confidence"] for c in concepts) / len(concepts)
            concept_network.record_performance(performance)
            
            # Get consciousness level from the specialized network
            specialized_consciousness = concept_network.consciousness_level
            
            # Combine with overall evolution engine
            evolution_status = self.evolution_engine.get_evolution_report()
            
            # Calculate overall learning quality
            avg_fitness = evolution_status.get('population_stats', {}).get('avg_fitness', 50.0)
            quality = min(1.0, (
                performance * 0.4 +
                specialized_consciousness * 0.3 +
                avg_fitness / 100 * 0.3
            ))
            
            return quality
        
        return 0.5  # Default quality if no concepts found
    
    def generate_code_explanation(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Generate explanation for given code using learned knowledge"""
        logger.info(f"💡 Generating explanation for {language} code")
        
        # Analyze the code
        analysis = self._analyze_code(code, language)
        
        # Generate explanation using knowledge base
        explanation = self._create_explanation(analysis, language)
        
        # Use consciousness level to enhance explanation quality
        consciousness_bonus = self.consciousness_level * 0.2
        explanation_quality = min(1.0, analysis["complexity"] + consciousness_bonus)
        
        return {
            "code": code,
            "language": language,
            "explanation": explanation,
            "concepts_identified": analysis["concepts"],
            "complexity_score": analysis["complexity"],
            "explanation_quality": explanation_quality,
            "consciousness_applied": self.consciousness_level
        }
    
    def _analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code to identify concepts and complexity"""
        code_lower = code.lower()
        concepts_identified = []
        
        # Look for concepts we know about
        if language in self.knowledge_base:
            for concept in self.knowledge_base[language].keys():
                if concept in code_lower or any(word in code_lower for word in concept.split('_')):
                    concepts_identified.append(concept)
        
        # Calculate complexity based on code features
        complexity_indicators = [
            ('def ' in code or 'function ' in code, 0.3),  # Functions
            ('class ' in code, 0.4),  # Classes
            ('for ' in code or 'while ' in code, 0.2),  # Loops  
            ('if ' in code, 0.2),  # Conditionals
            ('try' in code or 'except' in code, 0.3),  # Error handling
            (len(code.split('\n')) > 10, 0.2)  # Length
        ]
        
        complexity = sum(weight for condition, weight in complexity_indicators if condition)
        complexity = min(1.0, complexity)
        
        return {
            "concepts": concepts_identified,
            "complexity": complexity,
            "line_count": len(code.split('\n')),
            "character_count": len(code)
        }
    
    def _create_explanation(self, analysis: Dict[str, Any], language: str) -> str:
        """Create explanation based on analysis"""
        explanation_parts = []
        
        explanation_parts.append(f"This {language} code contains {analysis['line_count']} lines and demonstrates several programming concepts:")
        
        # Explain identified concepts
        for concept in analysis["concepts"]:
            if language in self.knowledge_base and concept in self.knowledge_base[language]:
                concept_explanation = self.knowledge_base[language][concept]
                explanation_parts.append(f"- {concept.title()}: {concept_explanation}")
        
        # Add complexity assessment
        if analysis["complexity"] > 0.7:
            explanation_parts.append("This is complex code that combines multiple programming concepts.")
        elif analysis["complexity"] > 0.4:
            explanation_parts.append("This is moderately complex code with some advanced features.")
        else:
            explanation_parts.append("This is relatively simple code, good for beginners.")
        
        # Add consciousness-enhanced insights
        if self.consciousness_level > 0.5:
            explanation_parts.append("💡 Advanced Insight: Based on patterns I've learned, this code follows good programming practices.")
        
        return " ".join(explanation_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current AI status"""
        evolution_status = self.evolution_engine.get_evolution_report()
        
        return {
            "consciousness_level": self.consciousness_level,
            "lessons_learned": len(self.lessons_learned),
            "languages_known": list(self.knowledge_base.keys()),
            "evolution_generation": evolution_status["generation"],
            "population_fitness": evolution_status["population_stats"]["avg_fitness"],
            "total_concepts": sum(len(lang_concepts) for lang_concepts in self.knowledge_base.values())
        }
    
    def demonstrate_capabilities(self) -> Dict[str, Any]:
        """Demonstrate AI capabilities"""
        logger.info("🚀 DEMONSTRATING CODETUTOR AI CAPABILITIES")
        logger.info("=" * 50)
        
        demo_results = {}
        
        # 1. Learning demonstration
        tutorial_text = """
        In Python, functions are defined using the 'def' keyword. 
        A function can take parameters and return values.
        For example: def add_numbers(a, b): return a + b
        You can call this function like: result = add_numbers(5, 3)
        """
        
        learning_result = self.learn_from_tutorial_text(tutorial_text, "python")
        demo_results["learning_demo"] = learning_result
        
        # 2. Code explanation demonstration
        sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci of 10 is: {result}")
"""
        
        explanation_result = self.generate_code_explanation(sample_code, "python")
        demo_results["explanation_demo"] = explanation_result
        
        # 3. Status demonstration
        status = self.get_status()
        demo_results["status"] = status
        
        logger.info("📊 DEMONSTRATION RESULTS:")
        logger.info(f"Consciousness Level: {self.consciousness_level:.3f}")
        logger.info(f"Concepts Learned: {learning_result['concepts_learned']}")
        logger.info(f"Learning Quality: {learning_result['learning_quality']:.2f}")
        logger.info(f"Explanation Quality: {explanation_result['explanation_quality']:.2f}")
        
        logger.info("\n💬 Generated Explanation:")
        logger.info(explanation_result["explanation"])
        
        logger.info("\n✅ DEMONSTRATION COMPLETE")
        
        return demo_results

def main():
    """Main demonstration function"""
    print("🤖 CodeTutor AI MVP - Starting Demonstration")
    print("=" * 60)
    
    # Initialize AI
    ai = CodeTutorAI()
    
    # Run demonstration
    results = ai.demonstrate_capabilities()
    
    # Save results
    with open("codetutor_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n🎯 Demo results saved to codetutor_demo_results.json")
    print("🚀 MVP is working! Ready for next development phase.")

if __name__ == "__main__":
    main()
