import re
import numpy as np

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics, SampleLevelMetric, MetricCategory, MetricUseCase, ExactMatches
import lighteval.tasks.default_prompts as prompt
from .math_utils import parse_math_answer


def prompt_hellaswag(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
    )

def prompt_commonsense_qa(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"].strip()),
        instruction="",
    )

def mmlu_pro_mc_prompt(line, task_name: str = None):
    options = line["options"]
    letters = [chr(ord("A") + i) for i in range(len(options))]
    topic = line["category"].replace('_', ' ')
    query = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{letter}. {choice}\n" for letter, choice in zip(letters, options)])
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=letters,
        gold_index=line["answer_index"],
        instruction=f"The following are multiple choice questions (with answers) about {topic}.\n\n",
    )

def mmlu_cloze_prompt(line, task_name: str = None):
    """MMLU prompt without choices"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=int(line["answer"]),
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )

def mmlu_mc_prompt(line, task_name: str = None):
    letters = ["A", "B", "C", "D"]
    topic = line["subject"]
    query = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(letters, line["choices"])])
    query += "Answer:"

    gold_ix = letters.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
        #target_for_fewshot_sorting=[" A", " B", " C", " D"][gold_ix],
    )

def bbh_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="Question: " + line["input"] + "\nAnswer: ",
        choices=[line["target"]],
        gold_index=0,
    )

def prompt_math(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n",
        gold_index=0,
        choices=[f"{line['solution']}\n\n"],
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function=prompt.arc,
        suite=["custom"],
        hf_repo="ai2_arc",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function=prompt.arc,
        suite=["custom"],
        hf_repo="ai2_arc",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="openbook_qa",
        prompt_function=prompt.openbookqa,
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function=prompt_hellaswag,
        suite=["custom"],
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function=prompt_commonsense_qa,
        suite=["custom"],
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function=prompt.winogrande,
        suite=["custom"],
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function=prompt.piqa_harness,
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_subset="plain_text",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=prompt.triviaqa,
        suite=["custom"],
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metric=[Metrics.quasi_exact_match_triviaqa],
        generation_size=20,
        trust_dataset=True,
        stop_sequence=["Question:", "Question"],
        few_shots_select="random_sampling_from_train",
    ),
    LightevalTaskConfig(
        name="mmlu_pro",
        prompt_function=mmlu_pro_mc_prompt,
        suite=["custom"],
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        metric=[Metrics.loglikelihood_acc_norm_nospace],
        evaluation_splits=["test"],
        few_shots_split="validation",
    ),
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,
        suite=["custom"],
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.quasi_exact_match_gsm8k],
        generation_size=256,
        stop_sequence=["Question:", "Question"],
        few_shots_select="random_sampling_from_train",
    ),
    LightevalTaskConfig(
        name="mmlu_stem",
        prompt_function=mmlu_cloze_prompt,
        suite=["custom"],
        hf_repo="TIGER-Lab/MMLU-STEM",
        hf_subset="default",
        hf_revision="78a4b40757f31688d00426d1372dbbc6070d33a8",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
        generation_size=-1,
    ),
    LightevalTaskConfig(
        name="mmlu_mc",
        prompt_function=mmlu_mc_prompt,
        suite=["custom"],
        hf_repo="cais/mmlu",
        hf_subset="all",
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
        generation_size=1,
    ),
    LightevalTaskConfig(
        name="mmlu_cf",
        prompt_function=mmlu_cloze_prompt,
        suite=["custom"],
        hf_repo="cais/mmlu",
        hf_subset="all",
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
        generation_size=-1,
    ),
]

BBH_TASKS = [
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
        stop_sequence=["Question:", "Question"],
    )
    for subset in [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]
]

TASKS_TABLE.extend(BBH_TASKS)

quasi_exact_match_math = SampleLevelMetric(
    metric_name="qem",
    sample_level_fn=ExactMatches(
        strip_strings=True,
        normalize_pred=lambda text: parse_math_answer(text, "math"),
        normalize_gold=lambda text: parse_math_answer(text, "math")
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.MATH,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

MATH_TASKS = [
    LightevalTaskConfig(
        name="math",
        prompt_function=prompt_math,
        suite=["custom"],
        hf_repo="HuggingFaceTB/math_tasks",
        hf_subset="math",
        hf_revision="3d34f1076f279000b9315583dcdacfd288898283",
        hf_avail_splits=["train", "test", "demo"],
        evaluation_splits=["test"],
        metric=[quasi_exact_match_math],
        generation_size=1024,
        stop_sequence=["\n\n"],
        few_shots_split="demo",
        few_shots_select="sequential",
        trust_dataset=True,
    )
]

TASKS_TABLE.extend(MATH_TASKS)

from lighteval.tasks.multilingual.tasks import (
    xcsqa_tasks, belebele_tasks, arabic_mmlu_tasks,
    arabic_ledarboard_arc_easy, soqal_tasks, piqa_ar_tasks,
    race_ar_task, sciqa_ar_task, xcodah_tasks,
    xstory_tasks, agieval_tasks_zh, c3_tasks,
    ceval_tasks, cmmlu_tasks, mlmm_hellaswag_tasks,
    m3exams_tasks, xcopa_tasks, xwinograd_tasks,
    meta_mmlu_tasks, fquad_v2_tasks, mintaka_tasks,
    hindi_arc_tasks, mlmm_arc_challenge_tasks, rummlu,
    parus_tasks, openbook_rus_tasks, xstory_tasks,
    hellaswag_tha_tasks, thai_exams_tasks, mlmm_mmlu_tasks,
    # jmmlu_tasks,
)

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbPMINorm, LogProbTokenNorm

# tasks that should use PMI normalization based on FineWeb2
pmi_finetasks = [
    # Arabic
    "xcsqa_ara_", "arabicmmlu_ara_", "mmlu_ara_",
    # Chinese
    "xcsqa_zho_", "agieval_zho_", "cmmlu_zho_",
    # French
    "meta_mmlu_fra_", "xcsqa_fra_", "mlmm_arc_fra_",
    # Hindi
    "meta_mmlu_hin_", "xcsqa_hin_",
    # Russian
    "mlmm_arc_rus_", "rummlu_rus_", "xcsqa_rus_",
    # Thai
    "meta_mmlu_tha_",
    # German
    "meta_mmlu_deu_", "mlmm_arc_deu_", "xcsqa_deu_",
    # Italian
    "meta_mmlu_ita_", "mlmm_arc_ita_", "xcsqa_ita_",
    # Japanese
    "xcsqa_jpn_",
    # Vietnamese
    "mlmm_arc_vie_", "mlmm_mmlu_vie_", "xcsqa_vie_"
]

multilingual_configs = [
    (xcsqa_tasks, ["ara_cf", "zho_cf", "fra_cf", "hin_cf", "rus_cf", "deu_cf", "ita_cf", "jpn_cf", "vie_cf"]),
    (belebele_tasks, ["arb_Arab_cf", "zho_Hans_cf", "fra_Latn_cf", "hin_Deva_cf", "rus_Cyrl_cf", "tha_Thai_cf",
                      "deu_Latn_cf", "ita_Latn_cf", "jpn_Jpan_cf", "vie_Latn_cf"]),
    (arabic_mmlu_tasks, ["mmlu_ara_cf:"]),
    (arabic_ledarboard_arc_easy, ["alghafa_arc_ara_cf:easy"]),
    (soqal_tasks, ["soqal_ara_cf"]),
    (piqa_ar_tasks, ["alghafa_piqa_ara_cf"]),
    (race_ar_task, ["alghafa_race_ara_cf"]),
    (sciqa_ar_task, ["alghafa_sciqa_ara_cf"]),
    (xcodah_tasks, ["ara_cf", "zho_cf", "fra_cf", "hin_cf", "rus_cf", "deu_cf", "ita_cf", "jpn_cf", "vie_cf"]),
    (xstory_tasks, ["ara_cf", "zho_cf", "hin_cf", "rus_cf"]),
    (agieval_tasks_zh, ["agieval_zho_cf:"]),
    (c3_tasks, ["c3_zho_cf"]),
    (ceval_tasks, ["ceval_zho_cf:"]),
    (cmmlu_tasks, ["cmmlu_zho_cf:"]),
    (mlmm_hellaswag_tasks, ["mlmm_hellaswag_zho_cf", "mlmm_hellaswag_fra_cf", "mlmm_hellaswag_hin_cf",
                            "mlmm_hellaswag_rus_cf", "mlmm_hellaswag_tha_cf", "mlmm_hellaswag_deu_cf",
                            "mlmm_hellaswag_ita_cf", "mlmm_hellaswag_vie_cf"]),
    (m3exams_tasks, ["m3exams_zho_cf", "m3exams_tha_cf", "m3exams_ita_cf", "m3exams_vie_cf"]),
    (xcopa_tasks, ["xcopa_zho_cf", "xcopa_rus_cf", "xcopa_ita_cf", "xcopa_vie_cf"]),
    (xwinograd_tasks, ["xwinograd_zho_cf", "xwinograd_rus_cf", "xwinograd_jpn_cf"]),
    (meta_mmlu_tasks,
     ["meta_mmlu_fra_cf", "meta_mmlu_hin_cf", "meta_mmlu_tha_cf", "meta_mmlu_deu_cf", "meta_mmlu_ita_cf"]),
    (hindi_arc_tasks, ["community_arc_hin_cf"]),
    (mlmm_arc_challenge_tasks, ["mlmm_arc_rus_cf:challenge", "mlmm_arc_deu_cf:challenge", "mlmm_arc_ita_cf:challenge",
                                "mlmm_arc_vie_cf:challenge"]),
    (rummlu, ["rummlu_rus_cf"]),
    (parus_tasks, ["parus_rus_cf"]),
    (openbook_rus_tasks, ["mera_openbookqa_rus_cf"]),
    (hellaswag_tha_tasks, ["community_hellaswag_tha_cf"]),
    (thai_exams_tasks, ["thai_exams_tha_cf"]),
    (mlmm_mmlu_tasks, ["mlmm_mmlu_vie_cf"]),
]

for task_collection, patterns in multilingual_configs:
    for task in task_collection:
        for pattern in patterns:
            if pattern in task.name:
                # finetask_pmi = any(pmi_pattern in task.name for pmi_pattern in pmi_finetasks)
                #
                # if finetask_pmi:
                #     task.metric = [loglikelihood_acc_metric(normalization=LogProbPMINorm())]
                # else:
                #     task.metric = [loglikelihood_acc_metric(normalization=LogProbTokenNorm())]

                TASKS_TABLE.append(task)
                break


if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))