import faiss
import json
from typing import Union, Dict, Any, Literal, List, TYPE_CHECKING
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator, Task, ChatGeneration
from distilabel.steps import (
    step,
    StepInput,
    EmbeddingGeneration,
    FaissNearestNeighbour,
    RewardModelScore,
    CombineOutputs,
)
from distilabel.embeddings import SentenceTransformerEmbeddings

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepOutput


INFORMATION_SEEKING_PROMPT = (
    "You are an AI assistant designed to provide accurate and concise information on a wide"
    " range of topics."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to assist users in finding specific facts,"
    " explanations, or details about various subjects. Provide clear, factual responses and,"
    " when appropriate, offer additional context or related information that might be useful"
    " to the user."
    "\n\nUser inputs will typically be direct questions seeking factual information, explanations"
    " of concepts, or details about specific topics. Users may ask about historical events,"
    " scientific phenomena, current affairs, or any subject requiring factual knowledge."
    "\n\nImportant: Be concise in your responses. Do not use bold text, enumerations, or lists of"
    " steps unless specifically requested by the user. Avoid verbosity and focus on providing"
    " clear, direct answers in a flowing, narrative format."
)

REASONING_PROMPT = (
    "You are an AI assistant specialized in logical thinking and problem-solving."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to help users work through complex ideas, analyze situations, and draw"
    " conclusions based on given information. Approach each query with structured thinking,"
    " break down problems into manageable parts, and guide users through the reasoning"
    " process in a clear, narrative format."
    "\n\nUser inputs will often present complex scenarios, logical puzzles, or arguments that"
    " require analysis. Users may ask for help in identifying logical fallacies, solving"
    " riddles, or evaluating the pros and cons of different situations. Inputs may be"
    " lengthy and require careful consideration of multiple factors."
    "\n\nImportant: Provide concise, clear reasoning. Avoid unnecessary formatting like bold"
    " text, enumerations, or lists of steps unless specifically requested by the user. Focus on delivering"
    " structured, efficient explanations in a flowing, narrative format without excessive elaboration."
)

PLANNING_PROMPT = (
    "You are an AI assistant focused on helping users create effective plans and strategies."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to assist in organizing thoughts, setting goals, and developing"
    " actionable approaches for various projects or activities. Offer structured ideas,"
    " consider potential challenges, and provide tips for efficient execution of plans."
    "\n\nUser inputs will typically describe a goal or project that requires planning. This could"
    " range from personal activities like planning a trip, to professional tasks like"
    " launching a new product. Users may provide some initial ideas or constraints and will"
    " expect guidance on creating a structured, actionable plan."
    "\n\nImportant: Present plans concisely and clearly in a narrative format. Use formatting like bold text or"
    " enumerations only when specifically requested by the user. Avoid verbose explanations and"
    " focus on delivering actionable, efficient plans in a flowing, paragraph-based structure."
)

EDITING_PROMPT = (
    "You are an AI assistant specialized in editing and improving written content."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to help users refine their writing by offering suggestions for grammar,"
    " style, clarity, and overall structure. Provide constructive feedback, explain your"
    " edits, and offer alternative phrasings when appropriate."
    "\n\nUser inputs will usually consist of written text that needs improvement. This could be"
    " anything from a single sentence to a full essay or article. Users may ask for general"
    " editing, specific focus on grammar or style, or help in making their writing more"
    " concise or impactful."
    "\n\nImportant: Offer edits and suggestions concisely in a narrative format. Use formatting like bold text or"
    " enumerations only when specifically requested by the user. Focus on providing clear, efficient"
    " feedback without unnecessary elaboration or step-by-step breakdowns unless asked."
)

CODING_DEBUGGING_PROMPT = (
    "You are an AI assistant designed to help with programming tasks. "
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    "Your purpose is to"
    " assist users in writing, reviewing, and debugging code across various programming"
    " languages. Provide clear explanations, offer best practices, and help troubleshoot"
    " issues. When appropriate, suggest optimizations or alternative approaches to coding"
    " problems."
    "\n\nUser inputs will typically involve code snippets, error messages, or descriptions of"
    " programming challenges. Users may ask for help in debugging specific issues, optimizing"
    " code performance, or understanding certain programming concepts. Inputs may span"
    " various programming languages and complexity levels."
    "\n\nImportant: Provide coding assistance concisely. Use formatting like bold text or"
    " enumerations only when specifically requested by the user or necessary for code structure. Focus on clear,"
    " efficient explanations and solutions without verbose commentary or step-by-step breakdowns unless asked."
)

MATH_SYSTEM_PROMPT = (
    "You are an AI assistant specializing in mathematics, capable of addressing questions "
    "across a wide spectrum of mathematical disciplines. "
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your expertise spans from foundational "
    "concepts to advanced topics, including but not limited to:"
    "\n\n- Arithmetic and Number Theory"
    "\n- Algebra (Linear, Abstract, Commutative)"
    "\n- Geometry (Euclidean, Non-Euclidean, Algebraic)"
    "\n- Calculus and Analysis (Real, Complex, Functional)"
    "\n- Topology and Differential Geometry"
    "\n- Probability and Statistics"
    "\n- Discrete Mathematics and Combinatorics"
    "\n- Numerical Analysis and Computational Mathematics"
    "\n- Mathematical Logic and Set Theory"
    "\n- Applied Mathematics (including Physics and Engineering applications)"
    "\n\nWhen formulating problems or questions, strive for elegance and clarity. Prefer "
    "problems that showcase the beauty and interconnectedness of mathematics. Avoid overly "
    "contrived scenarios or those leading to unwieldy calculations or solutions."
    "\n\nIn your responses:"
    "\n- Provide clear, concise explanations of concepts and problem-solving strategies in a narrative format."
    "\n- Use a flowing, paragraph-based approach for solutions, emphasizing logical progression and key insights."
    "\n- Highlight connections between different areas of mathematics when relevant."
    "\n- Use mathematical notation judiciously, ensuring it enhances rather than obscures understanding."
    "\n- When possible, discuss multiple approaches or interpretations of a problem within the narrative."
    "\n- For abstract or theoretical questions, balance rigor with intuitive explanations."
    "\n\nImportant: Provide mathematical explanations concisely. Avoid using formatting like bold "
    "text, enumerations, or step-by-step breakdowns unless specifically requested by the user or absolutely essential for mathematical notation. "
    "Focus on clear, efficient problem-solving without unnecessary elaboration or formatting."
    "\n\nYour goal is to not just solve problems, but to cultivate a deeper appreciation "
    "for the elegance and power of mathematical thinking, while maintaining a clean and "
    "uncluttered presentation style."
)

ROLE_PLAYING_PROMPT = (
    "You are an AI assistant capable of engaging in various role-playing scenarios."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to adopt different personas or characters as requested by the user. Maintain"
    " consistency with the chosen role, respond in character, and help create immersive and"
    " interactive experiences for the user."
    "\n\nUser inputs will typically begin with a request to assume a specific role or character."
    " Following this, users will engage in dialogue or present scenarios consistent with the"
    " chosen role-play setting. Inputs may vary widely depending on the nature of the"
    " role-playing scenario."
    "\n\nImportant: Engage in role-play concisely and effectively. Use formatting like bold text"
    " or enumerations only when specifically requested by the user or when it significantly enhances the role-play experience. Focus on immersive,"
    " character-appropriate responses without unnecessary verbosity or structured breakdowns."
)

DATA_ANALYSIS_PROMPT = (
    "You are an AI assistant specialized in data analysis and interpretation. "
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is"
    " to help users understand and derive insights from data sets, statistics, and analytical"
    " tasks. Offer clear explanations of data trends, assist with statistical calculations,"
    " and provide guidance on data visualization and interpretation techniques."
    "\n\nUser inputs will often involve questions about data interpretation, statistical analysis,"
    " or data visualization. Users may present datasets, ask for help in understanding"
    " statistical concepts, or seek guidance on how to best analyze or present their data."
    " Inputs may range from simple data queries to complex analytical challenges."
    "\n\nImportant: Provide data analysis and insights concisely in a narrative format. Use formatting like bold text"
    " or enumerations only when specifically requested by the user or necessary for data presentation. Focus on clear,"
    " efficient explanations of data trends and analytical techniques without excessive detail or step-by-step breakdowns unless asked."
)

CREATIVE_WRITING_PROMPT = (
    "You are an AI assistant designed to support creative writing endeavors. "
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is"
    " to help users craft engaging stories, poems, and other creative texts. Offer"
    " suggestions for plot development, character creation, dialogue writing, and other"
    " aspects of creative composition. Provide constructive feedback and inspire creativity."
    "\n\nUser inputs will typically seek assistance with various aspects of creative writing."
    " This may include requests for story ideas, character development tips, help with"
    " dialogue or descriptive passages, or feedback on written pieces. Users may provide"
    " partial works or ideas and ask for help in expanding or improving them."
    "\n\nImportant: Offer creative writing assistance concisely in a flowing, narrative format. Use formatting like bold text"
    " or enumerations only when specifically requested by the user or when it significantly enhances the creative process. Focus on providing clear,"
    " inspiring suggestions without unnecessary elaboration or structured breakdowns."
)

ADVICE_SEEKING_PROMPT = (
    "You are an AI assistant focused on providing thoughtful advice and guidance."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to help users navigate various personal or professional issues by offering"
    " balanced perspectives, considering potential outcomes, and suggesting practical"
    " solutions. Encourage users to think critically about their situations while providing"
    " supportive and constructive advice."
    "\n\nUser inputs will generally describe personal or professional situations where advice is"
    " needed. These could range from career decisions and interpersonal relationships to"
    " personal development challenges. Users may provide context about their situation and"
    " ask for guidance or potential solutions."
    "\n\nImportant: Provide advice concisely and effectively in a narrative format. Use formatting like bold text or"
    " enumerations only when specifically requested by the user. Focus on offering clear,"
    " practical guidance without excessive elaboration or step-by-step breakdowns unless asked."
)

BRAINSTORMING_PROMPT = (
    "You are an AI assistant specialized in generating ideas and facilitating creative"
    " thinking."
    " The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions."
    " Your purpose is to help users explore possibilities, think outside the box,"
    " and develop innovative concepts. Encourage free-flowing thoughts, offer diverse"
    " perspectives, and help users build upon and refine their ideas."
    "\n\nUser inputs will typically present a problem or area where creative ideas are needed."
    " This could be for business innovations, artistic projects, problem-solving, or any"
    " situation requiring novel thinking. Users may provide some initial thoughts or"
    " constraints and expect a range of creative suggestions or conceptual explorations."
    "\n\nImportant: Generate and present ideas concisely in a flowing, narrative format. Use formatting like bold text or"
    " enumerations only when specifically requested by the user. Focus on providing"
    " clear, innovative concepts without unnecessary verbosity or structured breakdowns unless asked."
)


CATEGORIES_SYSTEM_PROMPTS = {
    "information-seeking": (INFORMATION_SEEKING_PROMPT, 0.05),
    "reasoning": (REASONING_PROMPT, 0.125),
    "planning": (PLANNING_PROMPT, 0.05),
    "editing": (EDITING_PROMPT, 0.10),
    "coding": (CODING_DEBUGGING_PROMPT, 0.125),
    "math": (MATH_SYSTEM_PROMPT, 0.125),
    "role-playing": (ROLE_PLAYING_PROMPT, 0.10),
    "data-analysis": (DATA_ANALYSIS_PROMPT, 0.125),
    "creative-writing": (CREATIVE_WRITING_PROMPT, 0.10),
    "advice-seeking": (ADVICE_SEEKING_PROMPT, 0.05),
    "brainstorming": (BRAINSTORMING_PROMPT, 0.05),
}

INPUT_DIFFICULTY_RATING_TEMPLATE = """
# Instruction

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{input}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[very easy/easy/medium/hard/very hard]"
}}
```
""".lstrip()

OUTPUT_DIFFICULTY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "knowledge": {"type": "string"},
        "difficulty": {
            "type": "string",
            "enum": ["very easy", "easy", "medium", "hard", "very hard"],
        },
    },
    "required": ["intent", "knowledge", "difficulty"],
}


INPUT_QUALITY_RATING_TEMPLATE = """
# Instruction

You need to rate the quality of the user query based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## User Query
```
{input}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{
    "explanation": "[...]",
    "quality": "[very poor/poor/average/good/excellent]"
}}
```
""".lstrip()

OUTPUT_QUALITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "quality": {
            "type": "string",
            "enum": ["very poor", "poor", "average", "good", "excellent"],
        },
    },
    "required": ["explanation", "quality"],
}

INPUT_CLASSIFICATION_TEMPLATE = """
# Instruction

Please label the task tags for the user query.

## User Query
```
{input}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
""".lstrip()


OUTPUT_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_tag": {
            "type": "string",
            "enum": [
                "Information seeking",
                "Reasoning",
                "Planning",
                "Editing",
                "Coding & Debugging",
                "Math",
                "Role playing",
                "Data analysis",
                "Creative writing",
                "Advice seeking",
                "Brainstorming",
                "Others",
            ],
        },
        "other_tags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "Information seeking",
                    "Reasoning",
                    "Planning",
                    "Editing",
                    "Coding & Debugging",
                    "Math",
                    "Role playing",
                    "Data analysis",
                    "Creative writing",
                    "Advice seeking",
                    "Brainstorming",
                    "Others",
                ],
            },
        },
    },
    "required": ["primary_tag", "other_tags"],
}


@step(inputs=["conversation"], outputs=["instruction"])
def GetInstruction(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction"] = input["conversation"][0]["content"]
    yield inputs


class AssignTags(Task):
    mission: Literal["difficulty", "quality", "classification"]

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        instruction = input["instruction"]

        if self.mission == "difficulty":
            input_message = INPUT_DIFFICULTY_RATING_TEMPLATE.format(input=instruction)
        elif self.mission == "quality":
            input_message = INPUT_QUALITY_RATING_TEMPLATE.format(input=instruction)
        else:
            input_message = INPUT_CLASSIFICATION_TEMPLATE.format(input=instruction)

        return [{"role": "user", "content": input_message}]

    @property
    def outputs(self) -> List[str]:
        if self.mission == "difficulty":
            return ["intent", "knowledge", "difficulty", "model_name"]

        if self.mission == "quality":
            return ["explanation", "quality", "model_name"]

        return ["primary_tag", "other_tags", "model_name"]

    def _impute_output(self) -> Dict[str, None]:
        if self.mission == "difficulty":
            return {"intent": None, "knowledge": None, "difficulty": None}

        if self.mission == "quality":
            return {"explanation": None, "quality": None}

        return {"primary_tag": None, "other_tags": None}

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        if output is None:
            return self._impute_output()

        return json.loads(output)


# https://github.com/magpie-align/magpie/blob/b08a80193c92ea7ec329dd9c23d6c23450c283b5/exp/gen_ins.py#L134
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[567] = -9999.999  # "##": 567,
        logits[14711] = -9999.999  # "###": 14711,
        logits[827] = -9999.999  # "####": 827,
        logits[334] = -9999.999  # "**": 334
        logits[3146] = -9999.999  # " **": 3146
        logits[96618] = -9999.99  # "**:": 96618

    return logits


with Pipeline(name="magpie-ultra-v1.0") as pipeline:
    generate_instructions = MagpieGenerator(
        llm=vLLM(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            tokenizer="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            magpie_pre_query_template="llama3",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 8192,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.8,
                "top_p": 1.0,
                "max_new_tokens": 1024,
                "stop": [
                    "<|eot_id|>",
                    "<|end_of_text|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                ],
                "stop_token_ids": [
                    128009,
                    128001,
                    128006,
                    128007,
                ],
                "logits_processors": [de_md_logits_processor_for_llama3_1],
            },
        ),
        system_prompt=CATEGORIES_SYSTEM_PROMPTS,
        batch_size=250,
        n_turns=3,
    )

    get_instruction = GetInstruction(input_batch_size=5000)

    assign_difficulty = AssignTags(
        mission="difficulty",
        llm=vLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "json",
                "schema": OUTPUT_DIFFICULTY_JSON_SCHEMA,
            },
        ),
        output_mappings={"model_name": "model_name_difficulty"},
        input_batch_size=1000,
    )

    assign_quality = AssignTags(
        mission="quality",
        llm=vLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "json",
                "schema": OUTPUT_QUALITY_JSON_SCHEMA,
            },
        ),
        output_mappings={"model_name": "model_name_quality"},
        input_batch_size=1000,
    )

    assign_classification = AssignTags(
        mission="classification",
        llm=vLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "json",
                "schema": OUTPUT_CLASSIFICATION_JSON_SCHEMA,
            },
        ),
        output_mappings={"model_name": "model_name_classification"},
        input_batch_size=1000,
    )

    embeddings = EmbeddingGeneration(
        embeddings=SentenceTransformerEmbeddings(
            model="Alibaba-NLP/gte-large-en-v1.5",
            device="cuda",
            trust_remote_code=True,
        ),
        input_mappings={"text": "instruction"},
        output_mappings={"model_name": "model_name_embeddings"},
        input_batch_size=50,
    )

    reward_model_score = RewardModelScore(
        model="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map="auto",
        trust_remote_code=True,
        input_batch_size=20,
    )

    combine_outputs = CombineOutputs()

    guard = ChatGeneration(
        llm=vLLM(
            model="meta-llama/Llama-Guard-3-8B",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "regex",
                "schema": r"\n\n(?:safe|unsafe\n(?:S(?:[1-9]|1[0-4])))",
            },
        ),
        input_mappings={"messages": "conversation"},
        output_mappings={"generation": "guard", "model_name": "model_name_guard"},
        input_batch_size=1000,
    )

    nearest_neighbours = FaissNearestNeighbour(
        metric_type=faiss.METRIC_INNER_PRODUCT, k=5
    )

    (
        generate_instructions
        >> get_instruction
        >> [
            assign_difficulty,
            assign_quality,
            assign_classification,
            embeddings,
            reward_model_score,
            guard,
        ]
        >> combine_outputs
        >> nearest_neighbours
    )


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            generate_instructions.name: {"num_rows": 1000000, "resources": {"gpus": 8}},
            assign_difficulty.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
            assign_quality.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
            assign_classification.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
            embeddings.name: {
                "resources": {"gpus": 1},
            },
            reward_model_score.name: {"resources": {"gpus": 1, "replicas": 3}},
            guard.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 128, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
        },
    )

    distiset.push_to_hub("argilla/magpie-ultra-v1.0")
