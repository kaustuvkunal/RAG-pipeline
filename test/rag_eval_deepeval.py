import os
import sys
from typing import List, Dict

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.models import OllamaModel
from deepeval.models import OllamaEmbeddingModel
from deepeval.synthesizer.config import ContextConstructionConfig

from src.config import load_config, RAGConfig
from src.pipeline.rag_pipeline import RAGPipeline
from src.llm_client import get_llm_client
from src.logging_config import setup_logging, get_logger
from src.retriever import get_retriever

setup_logging(level="INFO")

config: RAGConfig = load_config()
logger = get_logger("deepeval_evaluation")
logger.info("DeepEval Evaluation started")

#embedding_model = OllamaEmbeddingModel(model=config.embedding_model)

judge_llm = OllamaModel( model = config.llm_model, base_url="http://localhost:11434/", temperature=0)
 

def rag_eval_deepeval():
    logger.info("🧪 Starting DeepEval RAG Evaluation...")

    pipeline = RAGPipeline(config)
    chain = pipeline.get_chain()
    retriever = pipeline._retriever


    # Metrics
    logger.info("Loading RAG Triad metrics...")
    answer_relevancy = AnswerRelevancyMetric(threshold=config.answer_relevancy_threshold, model=judge_llm)
    faithfulness = FaithfulnessMetric(threshold=config.faithfulness_threshold, model=judge_llm)
    contextual_relevancy = ContextualRelevancyMetric(threshold=config.contextual_relevancy_threshold, model=judge_llm)

    metrics = [answer_relevancy, faithfulness, contextual_relevancy]

    # Synthetic Test Cases with Ollama Embedding (as in notebook)
    logger.info("🔄 Generating synthetic test cases from documents...")
    
    test_cases: List[LLMTestCase] = []


    try:

        embedding_model = OllamaEmbeddingModel(model=config.embedding_model)
         
        
        synthesizer = Synthesizer(model=judge_llm)
        
        context_config = ContextConstructionConfig(
            critic_model=judge_llm,
            embedder=embedding_model
        )
        # check if doc_path  contains single file or folders 
        if os.path.isdir(config.doc_path):
            from pathlib import Path
            files_list = [str(p) for p in Path(config.doc_path).glob("*.pdf")]
            logger.info(f"🔄 Generating synthetic test cases for - {files_list}  files")
        else:
            files_list = [config.doc_path]
            logger.info(f"🔄 Generating synthetic test cases for - {files_list}  file ")

        
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=files_list,
            context_construction_config=context_config
        )

        # Convert to test cases
        for golden in goldens:
            response = chain.invoke({"question": golden.input})
            
            # Robust retriever call - handles both old and new LangChain
            try:
                # Try standard method
                docs = retriever.get_relevant_documents(golden.input)
            except AttributeError:
                try:
                    # New LangChain style
                    result = retriever.invoke(golden.input)
                    docs = result if isinstance(result, list) else [result]
                except Exception as inner_e:
                    logger.warning(f"Retriever call failed: {inner_e}. Using empty context.")
                    docs = []
            
            retrieval_context = [doc.page_content for doc in docs 
                               if hasattr(doc, 'page_content') and doc.page_content.strip()]
            
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=response,
                retrieval_context=retrieval_context,
            )
            test_cases.append(test_case)

        logger.info(f"✅ Successfully generated {len(test_cases)} synthetic test cases.")

    except Exception as e:
        logger.error(f"❌ Failed to generate synthetic test cases: {e}")
        logger.error("Make sure Ollama is running with 'nomic-embed-text' model pulled.")
        return []

    if not test_cases:
        logger.error("No test cases generated.")
        return []

    # Run Evaluation
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"🧪 Evaluating case {i+1}/{len(test_cases)}")

        try:
            scores = {}
            for metric in metrics:
                metric.measure(test_case)
                scores[metric.__class__.__name__] = metric.score

            results.append({
                "query": test_case.input,
                "response_preview": test_case.actual_output[:180] + "..." if len(test_case.actual_output) > 180 else test_case.actual_output,
                "retrieved_chunks": len(test_case.retrieval_context or []),
                "scores": scores
            })

            avg = sum(scores.values()) / len(scores) if scores else 0.0
            logger.info(f"✅ Case {i+1} completed | Avg Score: {avg:.3f}")

        except Exception as e:
            logger.error(f"❌ Error on case {i+1}: {e}")

    return results


def log_evaluation_results(results: List[Dict]):
    logger.info("=" * 90)
    logger.info(" " * 28 + "🧪 RAG DEEPEVAL EVALUATION RESULTS")
    logger.info("=" * 90)

    for r in results:
        if "error" in r:
            logger.error(f"❌ {r.get('query', '')[:70]}... → ERROR")
            continue

        avg = sum(r["scores"].values()) / len(r["scores"]) if r["scores"] else 0.0
        logger.info(f"📌 Query : {r['query']}")
        logger.info(f"   Chunks : {r['retrieved_chunks']} | Avg Score : {avg:.3f}")

        for name, score in r["scores"].items():
            logger.info(f"   {name.replace('Metric', ''):18}: {score:.3f}")

    logger.info("=" * 90)


if __name__ == "__main__":
    results = rag_eval_deepeval()
    log_evaluation_results(results)