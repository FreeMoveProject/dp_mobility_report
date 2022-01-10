import base64
import io


# TODO: better implementation and svg instead of png
def fig_to_html(fig):
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    html = '<div><img src="data:image/png;base64, {}"></div>'.format(
        base64.b64encode(img.getvalue()).decode("utf-8")
    )
    return html
