import os
from modelscope_studio.components.pro.chatbot import ChatbotActionConfig, ChatbotBotConfig, ChatbotUserConfig, ChatbotWelcomeConfig, ChatbotMarkdownConfig

# Env
is_cn =True # os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio'
# api_key = os.getenv('API_KEY')


def get_text(text: str, cn_text: str):
    if is_cn:
        return cn_text
    return text


# Save history in browser
save_history = True


# Chatbot Config
def markdown_config():
    return ChatbotMarkdownConfig(latex_delimiters=[
        {
            "left": "$$",
            "right": "$$",
            "display": True
        },
        {
            "left": "$",
            "right": "$",
            "display": False
        },
        {
            "left": "\\(",
            "right": "\\)",
            "display": False
        },
        {
            "left": "\\begin{equation}",
            "right": "\\end{equation}",
            "display": True
        },
        {
            "left": "\\begin{align}",
            "right": "\\end{align}",
            "display": True
        },
        {
            "left": "\\begin{alignat}",
            "right": "\\end{alignat}",
            "display": True
        },
        {
            "left": "\\begin{gather}",
            "right": "\\end{gather}",
            "display": True
        },
        {
            "left": "\\begin{CD}",
            "right": "\\end{CD}",
            "display": True
        },
        {
            "left": "\\[",
            "right": "\\]",
            "display": True
        },
    ])


def user_config(disabled_actions=None):
    return ChatbotUserConfig(
        class_names=dict(content="user-message-content"),
        actions=[
            "copy", "edit",
            ChatbotActionConfig(
                action="delete",
                popconfirm=dict(title=get_text("Delete the message", "删除消息"),
                                description=get_text(
                                    "Are you sure to delete this message?",
                                    "确认删除该消息？"),
                                okButtonProps=dict(danger=True)))
        ],
        disabled_actions=disabled_actions)


def bot_config(disabled_actions=None):
    return ChatbotBotConfig(actions=[
        "copy", "edit",
        ChatbotActionConfig(
            action="retry",
            popconfirm=dict(
                title=get_text("Regenerate the message", "重新生成消息"),
                description=get_text(
                    "Regenerate the message will also delete all subsequent messages.",
                    "重新生成消息会删除所有后续消息。"),
                okButtonProps=dict(danger=True))),
        ChatbotActionConfig(action="delete",
                            popconfirm=dict(
                                title=get_text("Delete the message", "删除消息"),
                                description=get_text(
                                    "Are you sure to delete this message?",
                                    "确认删除该消息？"),
                                okButtonProps=dict(danger=True)))
    ],
                            avatar="./assets/qwen.png",
                            disabled_actions=disabled_actions)


def welcome_config():
    return ChatbotWelcomeConfig(
        variant="borderless",
        icon="./assets/qwen.png",
        title=get_text("Hello, I'm Qwen3", "你好，我是 Qwen3"),
        description=get_text("Select a model and enter text to get started.",
                             "选择模型并输入文本，开始对话吧。"),
        prompts=dict(
            title=get_text("How can I help you today?", "有什么我能帮助你的吗?"),
            styles={
                "list": {
                    "width": '100%',
                },
                "item": {
                    "flex": 1,
                },
            },
            items=[{
                "label":
                get_text("📅 Make a plan", "📅 制定计划"),
                "children": [{
                    "description":
                    get_text("Help me with a plan to start a business",
                             "帮助我制定一个创业计划")
                }, {
                    "description":
                    get_text("Help me with a plan to achieve my goals",
                             "帮助我制定一个实现目标的计划")
                }, {
                    "description":
                    get_text("Help me with a plan for a successful interview",
                             "帮助我制定一个成功的面试计划")
                }]
            }, {
                "label":
                get_text("🖋 Help me write", "🖋 帮我写"),
                "children": [{
                    "description":
                    get_text("Help me write a story with a twist ending",
                             "帮助我写一个带有意外结局的故事")
                }, {
                    "description":
                    get_text("Help me write a blog post on mental health",
                             "帮助我写一篇关于心理健康的博客文章")
                }, {
                    "description":
                    get_text("Help me write a letter to my future self",
                             "帮助我写一封给未来自己的信")
                }]
            }]),
    )


DEFAULT_SUGGESTIONS = [{
    "label":
    get_text('Make a plan', '制定计划'),
    "value":
    get_text('Make a plan', '制定计划'),
    "children": [{
        "label":
        get_text("Start a business", "开始创业"),
        "value":
        get_text("Help me with a plan to start a business", "帮助我制定一个创业计划")
    }, {
        "label":
        get_text("Achieve my goals", "实现我的目标"),
        "value":
        get_text("Help me with a plan to achieve my goals", "帮助我制定一个实现目标的计划")
    }, {
        "label":
        get_text("Successful interview", "成功的面试"),
        "value":
        get_text("Help me with a plan for a successful interview",
                 "帮助我制定一个成功的面试计划")
    }]
}, {
    "label":
    get_text('Help me write', '帮我写'),
    "value":
    get_text("Help me write", '帮我写'),
    "children": [{
        "label":
        get_text("Story with a twist ending", "带有意外结局的故事"),
        "value":
        get_text("Help me write a story with a twist ending",
                 "帮助我写一个带有意外结局的故事")
    }, {
        "label":
        get_text("Blog post on mental health", "关于心理健康的博客文章"),
        "value":
        get_text("Help me write a blog post on mental health",
                 "帮助我写一篇关于心理健康的博客文章")
    }, {
        "label":
        get_text("Letter to my future self", "给未来自己的信"),
        "value":
        get_text("Help me write a letter to my future self", "帮助我写一封给未来自己的信")
    }]
}]

DEFAULT_SYS_PROMPT = "You are a helpful and harmless assistant."

MIN_THINKING_BUDGET = 1

MAX_THINKING_BUDGET = 38

DEFAULT_THINKING_BUDGET = 38

DEFAULT_MODEL = "./models/Qwen/Qwen3-0.6B"

MODEL_OPTIONS = [
    # {
    #     "label": get_text("Qwen3-235B-A22B", "通义千问3-235B-A22B"),
    #     "modelId": "Qwen/Qwen3-235B-A22B",
    #     "value": "qwen3-235b-a22b"
    # },
    # {
    #     "label": get_text("Qwen3-32B", "通义千问3-32B"),
    #     "modelId": "Qwen/Qwen3-32B",
    #     "value": "qwen3-32b"
    # },
    # {
    #     "label": get_text("Qwen3-30B-A3B", "通义千问3-30B-A3B"),
    #     "modelId": "Qwen/Qwen3-30B-A3B",
    #     "value": "qwen3-30b-a3b"
    # },
    # {
    #     "label": get_text("Qwen3-14B", "通义千问3-14B"),
    #     "modelId": "Qwen/Qwen3-14B",
    #     "value": "qwen3-14b"
    # },
    # {
    #     "label": get_text("Qwen3-8B", "通义千问3-8B"),
    #     "modelId": "Qwen/Qwen3-8B",
    #     "value": "qwen3-8b"
    # },
    # {
    #     "label": get_text("Qwen3-4B", "通义千问3-4B"),
    #     "modelId": "Qwen/Qwen3-4B",
    #     "value": "qwen3-4b"
    # },
    {
        "label": get_text("Qwen3-1.7B", "通义千问3-1.7B"),
        "modelId": "Qwen/Qwen3-1.7B",
        "value": "qwen3-1.7b"
    },
    {
        "label": get_text("Qwen3-0.6B", "通义千问3-0.6B"),
        "modelId": "Qwen/Qwen3-0.6B",
        "value": "./models/Qwen/Qwen3-0.6B"
    },
]

for model in MODEL_OPTIONS:
    model[
        "link"] = is_cn and f"https://modelscope.cn/models/{model['modelId']}" or f"https://huggingface.co/{model['modelId']}"

MODEL_OPTIONS_MAP = {model["value"]: model for model in MODEL_OPTIONS}

DEFAULT_LOCALE = 'zh_CN' if is_cn else 'en_US'

DEFAULT_THEME = {
    "token": {
        "colorPrimary": "#6A57FF",
    }
}

DEFAULT_SETTINGS = {
    "model": DEFAULT_MODEL,
    "sys_prompt": DEFAULT_SYS_PROMPT,
    "thinking_budget": DEFAULT_THINKING_BUDGET
}
