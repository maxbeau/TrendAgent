def main() -> None:
    from app.core.app import create_app

    app = create_app()
    return app


application = main()
