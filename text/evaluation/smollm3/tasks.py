# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval. We need to use custom tasks for these benchmarks because many of the pre-existing tasks in LightEval are designed for different configurations or base models and thus we must adapt the prompt and metrics to the zero-shot generative case.

Usage:

lighteval vllm "model_name=HuggingFaceTB/SmolLM3-3B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
    "custom|gsm_plus|0|0,custom|mixeval_hard|0|0" \
    --use-chat-template \
    --output-dir evals/ \
    --custom-tasks tasks.py \
    --save-details
"""
from functools import partial
import numpy as np

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import Metrics, MetricCategory
from lighteval.metrics.metrics_sample import (
    JudgeLLM,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.extended.mix_eval.main import (
    flow_judge_for_freeform_template,
    flow_judge_for_multichoice_template,
    mean_dv_5,
    mixeval_freeform_prompt,
    mixeval_multichoice_prompt,
    process_judge_response,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import winogrand_adapter
from lighteval.tasks.multilingual.tasks import TASKS_TABLE as ML_TASKS_TABLE
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language
from lighteval.utils.utils import remove_reasoning_tags

try:
    from .evalplus_utils import (
        HUMAN_EVAL_STOP_SEQUENCES,
        MBPP_STOP_SEQUENCES,
        evaluate_humaneval_plus,
        evaluate_mbpp_plus,
        humaneval_plus_query,
        mbpp_plus_query,
    )
except ImportError:
    from evalplus_utils import (
        HUMAN_EVAL_STOP_SEQUENCES,
        MBPP_STOP_SEQUENCES,
        evaluate_humaneval_plus,
        evaluate_mbpp_plus,
        humaneval_plus_query,
        mbpp_plus_query,
    )

TASKS_TABLE = []
TASKS_TABLE.extend(ML_TASKS_TABLE)

#------------------
# BASE MODEL EVALS
#------------------
qa_metrics = [
    loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
    loglikelihood_acc_metric(normalization=LogProbCharNorm()),
]
all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

# ARC tasks
arc_tasks = [
    LightevalTaskConfig(
        name=f"arc_{formulation.name.lower()}:{subset.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="allenai/ai2_arc",
        hf_subset=f"ARC-{subset}",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in ["Easy", "Challenge"]
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(arc_tasks)

# BoolQ task
boolq_task = LightevalTaskConfig(
    name="boolq_cf",
    prompt_function=get_boolq_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "answer": line["answer"],
            "context": line["passage"],
        },
        formulation=CFFormulation(),
    ),
    suite=("custom",),
    hf_repo="google/boolq",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=5,
    stop_sequence=["\n"],
    metric=get_metrics_for_formulation(CFFormulation(), qa_metrics),
)
TASKS_TABLE.append(boolq_task)

# HellaSwag tasks
hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{formulation.name.lower()}",
        suite=["custom"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.ENGLISH,
            adapter=lambda line: {
                "activity_label": line["activity_label"],
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        evaluation_splits=["validation"],
        hf_avail_splits=["train", "validation"],
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(hellaswag_tasks)

# CommonsenseQA tasks
commonsense_qa_tasks = [
    LightevalTaskConfig(
        name=f"commonsenseqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": line["choices"]["label"].index(line["answerKey"].strip()),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(commonsense_qa_tasks)

# OpenBookQA tasks
openbook_qa_tasks = [
    LightevalTaskConfig(
        name=f"openbookqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(openbook_qa_tasks)

# Winogrande tasks
winogrande_tasks = [
    LightevalTaskConfig(
        name=f"winogrande_{formulation.name.lower()}",
        suite=("custom",),
        prompt_function=get_continuation_prompt_function(
            Language.ENGLISH,
            partial(winogrand_adapter, Language.ENGLISH),
            formulation=formulation,
        ),
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        trust_dataset=True,
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=qa_metrics,
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(winogrande_tasks)

# PIQA tasks
piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["goal"],
                "choices": [line["sol1"], line["sol2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        hf_subset="plain_text",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(piqa_tasks)

# SIQA tasks
siqa_tasks = [
    LightevalTaskConfig(
        name=f"siqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [line["answerA"], line["answerB"], line["answerC"]],
                "gold_idx": int(line["label"]) - 1,
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/social_i_qa",
        hf_revision="53620e5841fb12b08e082485797e7021d3684ea2",
        hf_subset="default",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(siqa_tasks)

# MMLU tasks
# fmt: off
MMLU_SUBSETS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
    'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]
# fmt: on

mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="cais/mmlu",
        hf_subset=subset,
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in MMLU_SUBSETS
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_tasks)

# MMLU Pro tasks
mmlu_pro_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_pro_tasks)

# TriviaQA tasks
triviqa_tasks = [
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=prompt.triviaqa,
        suite=("custom",),
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        generation_size=256,
        stop_sequence=("\n",),
        metric=[Metrics.quasi_exact_match_triviaqa],
        few_shots_select="random_sampling_from_train",
    )
]
TASKS_TABLE.extend(triviqa_tasks)


# BBH tasks
def bbh_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="Question: " + line["input"] + "\nAnswer: ",
        choices=[line["target"]],
        gold_index=0,
    )


# fmt: off
BBH_SUBSETS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]
# fmt: on

bbh_tasks = [
    LightevalTaskConfig(
        name=f"bbh:{subset}",
        prompt_function=bbh_prompt,
        suite=["custom"],
        hf_repo="lighteval/big_bench_hard",
        hf_subset=subset,
        hf_revision="80610173426f05e6f1448f047e2db4840a7dd899",
        metric=[Metrics.exact_match],
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        trust_dataset=True,
        stop_sequence=["Question:"],
    )
    for subset in BBH_SUBSETS
]
TASKS_TABLE.extend(bbh_tasks)

# GSM8K tasks
gsm8k_tasks = [
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,
        suite=("custom",),
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.expr_gold_metric],
        generation_size=256,
        stop_sequence=["Question:"],
        few_shots_select="random_sampling_from_train",
    )
]
TASKS_TABLE.extend(gsm8k_tasks)

# HumanEval+ / MBPP+ (blog base-model code evals; not shipped by LightEval)
def humaneval_plus_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=humaneval_plus_query(line),
        choices=[line.get("canonical_solution") or ""],
        gold_index=0,
        specific={
            "prompt": line["prompt"],
            "test": line["test"],
            "entry_point": line["entry_point"],
        },
    )


def mbpp_plus_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=mbpp_plus_query(line),
        choices=[line.get("code") or ""],
        gold_index=0,
        specific={
            "test": line["test"],
            "test_imports": line.get("test_imports") or [],
        },
    )


def humaneval_plus_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    prediction = predictions[0] if predictions else ""
    spec = formatted_doc.specific or {}
    return float(
        evaluate_humaneval_plus(
            prompt=spec["prompt"],
            prediction=prediction,
            test=spec["test"],
            entry_point=spec["entry_point"],
        )
    )


def mbpp_plus_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    prediction = predictions[0] if predictions else ""
    spec = formatted_doc.specific or {}
    return float(
        evaluate_mbpp_plus(
            prediction=prediction,
            test=spec["test"],
            test_imports=spec.get("test_imports") or [],
        )
    )


humaneval_plus_pass_at_1 = SampleLevelMetric(
    metric_name="humaneval_plus_pass@1",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.CODE,
    sample_level_fn=humaneval_plus_metric,
    corpus_level_fn=np.mean,
)

mbpp_plus_pass_at_1 = SampleLevelMetric(
    metric_name="mbpp_plus_pass@1",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.CODE,
    sample_level_fn=mbpp_plus_metric,
    corpus_level_fn=np.mean,
)

humaneval_plus = LightevalTaskConfig(
    name="humaneval_plus",
    suite=["custom"],
    prompt_function=humaneval_plus_prompt,
    hf_repo="evalplus/humanevalplus",
    hf_subset="default",
    hf_revision="d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metric=[humaneval_plus_pass_at_1],
    stop_sequence=HUMAN_EVAL_STOP_SEQUENCES,
    trust_dataset=True,
    version=0,
)

mbpp_plus = LightevalTaskConfig(
    name="mbpp_plus",
    suite=["custom"],
    prompt_function=mbpp_plus_prompt,
    hf_repo="evalplus/mbppplus",
    hf_subset="default",
    hf_revision="b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metric=[mbpp_plus_pass_at_1],
    stop_sequence=MBPP_STOP_SEQUENCES,
    trust_dataset=True,
    version=0,
)

TASKS_TABLE.extend([humaneval_plus, mbpp_plus])

# MATH tasks
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

math_tasks = [
    LightevalTaskConfig(
        name=f"math_cot:{config}",
        suite=("custom",),
        prompt_function=prompt.math_cot,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=config,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=4096,
        metric=[latex_gold_metric],
        stop_sequence=["\n"],
        trust_dataset=True,
        version=0,
    )
    for config in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]
TASKS_TABLE.extend(math_tasks)

#---------------------
# INSTRUCT MODEL EVALS
#---------------------
REASONING_TAG_PAIRS = [
    ("<think>", "</think>"),
]

###########
# MIXEVAL #
###########
"""The main differences between this implementation and LightEval's is that:

- we don't use GPT-3.5-Turbo among the list of judges, but instead use flowaicom/Flow-Judge-v0.1.
- we strip out the reasoning block from the predictions before passing them to the judge.
- we left-truncate the predictions to fit within the 2048 token limit of the judge model.
- we specify max_tokens=6144 for the judge to fit within the max context size of 8192 tokens.
"""

MIXEVAL_EASY_TASKS_LIST = ",".join(["mixeval_easy:freeform", "mixeval_easy:multichoice"])
MIXEVAL_HARD_TASKS_LIST = ",".join(["mixeval_hard:freeform", "mixeval_hard:multichoice"])

MAX_INPUT_TOKENS = 2048  # LightEval judges have max_model_len=8192 and require space for a long judge prompt. We allow 2048 tokens for the prediction to fit within the limit.


class JudgeLLMMixEval(JudgeLLM):
    def compute(self, sample_ids: list[str], responses: list, formatted_docs: list[Doc], **kwargs) -> dict[str, float]:
        """
        Compute the score of a generative task using a llm as a judge.
        The generative task can be multiturn with 2 turns max, in that case, we
        return scores for turn 1 and 2. Also returns user_prompt and judgement
        which are ignored later by the aggregator.
        """
        questions = [formatted_doc.specific["question"] for formatted_doc in formatted_docs]
        options = [formatted_doc.choices for formatted_doc in formatted_docs]
        golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]

        predictions = []
        num_truncated = 0
        for response in responses:
            prediction_text = remove_reasoning_tags(response[0].result[0], tag_pairs=REASONING_TAG_PAIRS).strip()

            # Left-truncate the prediction to fit within the max input tokens for the judge model.
            if len(response[0].generated_tokens[0]) > MAX_INPUT_TOKENS:
                # One token is worth ~4 characters, so we estimate the number of characters to truncate.
                prediction_text = f"{prediction_text[-MAX_INPUT_TOKENS * 4 :]}"

            predictions.append(prediction_text)

        print(f"Number of predictions truncated to fit within {MAX_INPUT_TOKENS} tokens: {num_truncated}")  # noqa: T201

        scores, messages, judgements = self.judge.evaluate_answer_batch(questions, predictions, options, golds)

        metrics = []
        for i in range(len(sample_ids)):
            metrics.append(
                {
                    f"judge_score_{self.short_judge_name}": scores[i],
                    f"user_prompt_{self.short_judge_name}": messages[i],
                    f"judgement_{self.short_judge_name}": judgements[i],
                }
            )

        return metrics


llm_judge_mixeval_multichoice_flow_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_flow"],
    higher_is_better={"judge_score_flow": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_for_multichoice_template,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
        short_judge_name="flow",
        max_tokens=1024,  # Flow judge has a context length limit of 8192 tokens and 2048 are reserved for the input
    ).compute,
    corpus_level_fn={
        "judge_score_flow": np.mean,
    },
)


llm_judge_mixeval_freeform_flow_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_flow"],
    higher_is_better={"judge_score": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_for_freeform_template,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
        short_judge_name="flow",
        max_tokens=1024,  # Flow judge has a context length limit of 8192 tokens and 2048 are reserved for the input
    ).compute,
    corpus_level_fn={
        "judge_score_flow": mean_dv_5,
    },
)

mixeval_freeform_easy = LightevalTaskConfig(
    name="mixeval_easy:freeform",
    prompt_function=mixeval_freeform_prompt,
    suite=["custom"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval",
    metric=[llm_judge_mixeval_freeform_flow_judge],
    hf_avail_splits=["free_form"],
    evaluation_splits=["free_form"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,  # overridden at runtime by the generation parameters
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)

mixeval_multichoice_easy = LightevalTaskConfig(
    name="mixeval_easy:multichoice",
    prompt_function=mixeval_multichoice_prompt,
    suite=["custom"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval",
    metric=[llm_judge_mixeval_multichoice_flow_judge],
    hf_avail_splits=["multiple_choice"],
    evaluation_splits=["multiple_choice"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,  # overridden at runtime by the generation parameters
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)

mixeval_freeform_hard = LightevalTaskConfig(
    name="mixeval_hard:freeform",
    prompt_function=mixeval_freeform_prompt,
    suite=["custom"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval_Hard",
    metric=[llm_judge_mixeval_multichoice_flow_judge],
    hf_avail_splits=["free_form"],
    evaluation_splits=["free_form"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,  # overridden at runtime by the generation parameters
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)


mixeval_multichoice_hard = LightevalTaskConfig(
    name="mixeval_hard:multichoice",
    prompt_function=mixeval_multichoice_prompt,
    suite=["custom"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval_Hard",
    metric=[llm_judge_mixeval_multichoice_flow_judge],
    hf_avail_splits=["multiple_choice"],
    evaluation_splits=["multiple_choice"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,  # overridden at runtime by the generation parameters
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)

TASKS_TABLE.extend(
    [
        mixeval_multichoice_easy,
        mixeval_freeform_easy,
        mixeval_multichoice_hard,
        mixeval_freeform_hard,
    ]
)


###########
# GSMPlus #
###########
def gsm_plus_prompt(line, task_name: str = None):
    # Prompt template adapted from
    # - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
    # - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
    # Note that it is important to have the final answer in a box for math-verify to work correctly
    MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

    # Some prompts require critical thinking (around 1k/10k), we skip them as
    # they are a bit trickier to eval with regular text extraction.
    if line["perturbation_type"] == "critical thinking":
        return None

    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


gsm_plus = LightevalTaskConfig(
    name="gsm_plus",
    suite=["custom"],
    prompt_function=gsm_plus_prompt,
    hf_repo="qintongli/GSM-Plus",
    hf_subset="default",
    hf_avail_splits=["testmini"],
    evaluation_splits=["testmini"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metric=[
        Metrics.math_pass_at_1_1n,
    ],
    stop_sequence=None,
    trust_dataset=True,
    version=0,
)

TASKS_TABLE.append(gsm_plus)


# remove pmi_norm from all tasks to save on double inference
for task in TASKS_TABLE:
    task.metric = [
        metric
        for metric in task.metric
        if metric.category != MetricCategory.MULTICHOICE_PMI
    ]

if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
