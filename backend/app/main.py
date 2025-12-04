from dotenv import load_dotenv

from app.core.app import create_app


def main() -> None:
    load_dotenv()
    app = create_app()
    return app


application = main()
