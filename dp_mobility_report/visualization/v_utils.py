import base64
import io

from matplotlib.figure import Figure


def fig_to_html(fig: Figure) -> str:
    img = io.StringIO()
    fig.savefig(img, format="svg", bbox_inches="tight")
    img_string = img.getvalue()
    return img_string


def fig_to_html_as_png(fig: Figure) -> str:
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    img_string = base64.b64encode(img.getvalue()).decode("utf-8")
    html = f"<img src='data:image/png;base64,{img_string}'>"
    return html
