"""Inference-only map website using Gradio's temporary tunnel."""
import os,secrets

def launch(share=True,port=7860):
    import gradio as gr
    from . import api
    generated=not os.getenv('APP_PASSWORD')
    if generated:os.environ['APP_PASSWORD']=secrets.token_urlsafe(18)
    if not api.service.status()['model_ready']:raise RuntimeError('Restore the trained deployment ZIP before launching')
    with gr.Blocks(title='Cloud Atlas launcher') as demo:
        gr.Markdown('# Cloud Atlas\n[Open the map website](/map)\n\nThis session uses the saved model. It does not train.')
    demo.launch(share=share,auth=('admin',os.environ['APP_PASSWORD']),server_port=port,prevent_thread_lock=True,show_error=False)
    # Register the protected map and API on the same origin served by the tunnel.
    demo.app.include_router(api.app.router)
    url=demo.share_url or demo.local_url
    print('Map website:',url.rstrip('/')+'/map')
    print('Username: admin. Password: your APP_PASSWORD Kaggle Secret.')
    if generated: print('Temporary password:',os.environ['APP_PASSWORD'])
    return demo
