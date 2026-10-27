"""Temporary Kaggle-hosted web demo using the same satellite/model service."""
import os
import secrets
from datetime import date,timedelta
from urllib.parse import urlencode
import gradio as gr
from .service import RestorationService


def make_demo(service=None):
    service=service or RestorationService()
    def area(lat,lon,km):
        import math
        from .satellite import check_area
        if not all(math.isfinite(v) for v in [lat,lon,km]) or not .2<=km<=8 or abs(lat)>79: raise gr.Error('Choose valid coordinates and an area width of 0.2â€“8 km')
        dy=km/2/111.32; dx=dy/math.cos(math.radians(lat))
        bbox=[lon-dx,lat-dy,lon+dx,lat+dy]; check_area(bbox); return bbox
    def search(lat,lon,km,start,end):
        try:
            bbox=area(lat,lon,km); result=service.client.search(bbox,start,end)
            choices=list(dict.fromkeys(s['datetime'] for s in result['scenes']))
            return gr.update(choices=choices,value=choices[0] if choices else None),bbox,('No images found.' if not choices else f'{len(choices)} acquisition times found. Select one and restore.')
        except Exception as exc: raise gr.Error(str(exc)) from None
    def restore(bbox,acquisition):
        if not bbox or not acquisition: raise gr.Error('Find and select an acquisition first')
        try:
            metadata,folder=service.restore(bbox,acquisition)
            return str(folder/'original.png'),str(folder/'restored.png'),[str(folder/n) for n in ['original.png','restored.png','restored.tif','source_reflectance.tif','metadata.json']],metadata
        except Exception as exc: raise gr.Error(str(exc)) from None
    def preview(lat,lon,km):
        bbox=area(lat,lon,km)
        url='https://www.openstreetmap.org/export/embed.html?'+urlencode({'bbox':','.join(map(str,bbox)),'layer':'mapnik','marker':f'{lat},{lon}'})
        return f'<iframe title="Selected area preview" src="{url}" style="width:100%;height:300px;border:0;border-radius:12px"></iframe>'
    with gr.Blocks(title='Cloud Atlas',theme=gr.themes.Soft(primary_hue='teal')) as demo:
        gr.Markdown('# Cloud Atlas\nSelect a location and date to retrieve Sentinel-2 imagery and generate a restored estimate. No photo upload needed.')
        readiness=service.status()
        gr.Markdown('**Model:** '+('ready' if readiness['model_ready'] else 'not trained yet')+' Â· **Satellite access:** '+('configured' if readiness['satellite_ready'] else 'add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to Kaggle Secrets and rerun the setup cell'))
        with gr.Row():
            with gr.Column(scale=1):
                lat=gr.Number(value=18.5204,label='Latitude'); lon=gr.Number(value=73.8567,label='Longitude'); km=gr.Slider(.2,8,value=2,step=.2,label='Area width (km)')
                start=gr.Textbox(value=str(date.today()-timedelta(days=30)),label='Start date (YYYY-MM-DD)'); end=gr.Textbox(value=str(date.today()),label='End date (YYYY-MM-DD)')
                preview_button=gr.Button('Preview area'); search_button=gr.Button('Find satellite images',variant='primary')
                scenes=gr.Dropdown(label='Acquisition time'); restore_button=gr.Button('Restore selected image',variant='primary')
                status=gr.Textbox(label='Status',interactive=False); bbox=gr.State(None)
            with gr.Column(scale=2):
                map_view=gr.HTML(preview(18.5204,73.8567,2))
                gr.Markdown('Map is a location preview. Change coordinates to select a different area. Restored pixels are estimates, not verified observations beneath opaque clouds.')
        with gr.Row():
            original=gr.Image(label='Original observation',type='filepath'); restored=gr.Image(label='Restored estimate',type='filepath')
        files=gr.File(label='Download imagery and metadata',file_count='multiple'); metadata=gr.JSON(label='Acquisition and processing details')
        def reset_selection():
            return gr.update(choices=[],value=None),None,'Selection changed. Find satellite images again.'
        for control in [lat,lon,km,start,end]:
            control.change(reset_selection,outputs=[scenes,bbox,status])
        preview_button.click(preview,[lat,lon,km],map_view)
        search_button.click(search,[lat,lon,km,start,end],[scenes,bbox,status])
        restore_button.click(restore,[bbox,scenes],[original,restored,files,metadata])
    return demo


def launch():
    password=os.getenv('APP_PASSWORD') or secrets.token_urlsafe(18)
    demo=make_demo(); demo.queue(max_size=3,default_concurrency_limit=1)
    print('Demo login username: admin')
    if not os.getenv('APP_PASSWORD'): print('Temporary demo password:',password)
    demo.launch(share=True,auth=('admin',password),prevent_thread_lock=True,show_error=False,
                allowed_paths=[str(__import__('pathlib').Path('web_results').resolve())])
    return demo
