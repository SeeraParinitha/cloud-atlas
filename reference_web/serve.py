"""Run the portal independently of a Kaggle or Gradio session."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Cloud Atlas web portal')
    parser.add_argument('--parent', required=True, help='finetune/hsformer/best.pt')
    parser.add_argument('--checkpoint', required=True, help='triplet_run/best.pt')
    parser.add_argument('--output', default='temporal_web_results')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    from .auth import google_config
    try: google_config()
    except ValueError as exc: parser.error(str(exc))
    import uvicorn
    from .app import create_app
    from .service import WholeReferenceService
    service = WholeReferenceService(args.parent, args.checkpoint, output=args.output)
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
