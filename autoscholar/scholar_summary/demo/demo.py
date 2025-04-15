import logging
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    # Step 1. Convert paper(pdf) to markdown
    logger.info("Starting PDF to Markdown conversion...")
    pdf_file_path = "autoscholar/scholar_summary/demo/2503.08696.pdf"
    markdown_output_dir = Path("autoscholar/scholar_summary/demo/")
    markdown_file_path = markdown_output_dir / "2503.08696/2503.08696.md"
    image_file_list = sorted(markdown_output_dir.glob("*.jpeg"))
    latex_template_file = Path("autoscholar/scholar_summary/demo/template.tex")
    latex_output_file = Path(
        "autoscholar/scholar_summary/demo/2503.08696/demo_summary.tex"
    )

    conversion_command = ["marker_single", pdf_file_path]
    conversion_command.extend(["--output_dir", markdown_output_dir])

    subprocess.run(conversion_command, check=True)
    logger.info("PDF to Markdown conversion completed.")

    # Step 2. Generate LaTeX summary
    logger.info("Starting LaTeX summary generation...")
    model_instance = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0.0},
    )

    chat_agent = ChatAgent(model=model_instance)

    with open(markdown_file_path, "r", encoding="utf-8") as markdown_file:
        paper_content = markdown_file.read()

    with open(latex_template_file, "r", encoding="utf-8") as template_file:
        latex_template_content = template_file.read()

    image_file_names = "\n".join([f"- {img.name}" for img in image_file_list])

    prompt_text = f"""
    你是一位精通 LaTeX 的学术助手。以下是用户提供的内容：

    1. 一篇论文的内容（Markdown 格式）
    2. 图像文件名列表
    3. 一个 LaTeX 模板，其中部分内容需要填写

    你的任务是：
    - 阅读论文内容和图像文件名
    - 理解论文的主旨、方法和结果
    - 仅填写 LaTeX 模板中待填写的部分，保留其原有结构和格式
    - 内容应尽量紧凑，突出重点
    - 注意处理特殊字符，例如将 % 替换为 \%
    - 返回完整的 LaTeX 源码，不要包含多余的说明
    - 不要返回 ``` latex ... ```，直接返回完整的文件内容

    ----------------------
    [论文内容]
    {paper_content}

    ----------------------
    [图像文件]
    {image_file_names}

    ----------------------
    [LaTeX 模板]
    {latex_template_content}

    ----------------------
    请补全上述 LaTeX 模板并返回完整代码。
    """

    response = chat_agent.step(prompt_text)
    completed_latex_content = response.msgs[0].content

    with open(latex_output_file, "w", encoding="utf-8") as output_file:
        output_file.write(completed_latex_content)

    logger.info(f"LaTeX summary generated: {latex_output_file}")

    # Step 3. Compile LaTeX to PDF
    logger.info("Starting LaTeX compilation to PDF...")
    subprocess.run(
        [
            "lualatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(latex_output_file.parent.resolve()),
            str(latex_output_file.name),
        ],
        check=True,
        cwd=str(latex_output_file.parent.resolve()),
    )
    logger.info("LaTeX compilation completed. PDF generated.")

except subprocess.CalledProcessError as e:
    logger.error(f"Subprocess failed with error: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")
