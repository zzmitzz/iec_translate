import logging
import importlib.resources as resources
import base64
import os
import pathlib

logger = logging.getLogger(__name__)

def get_web_interface_html():
    """Loads the HTML for the web interface using importlib.resources."""
    try:
        # Get the current directory path
        current_dir = pathlib.Path(__file__).parent
        html_file = current_dir / 'live_transcription.html'
        
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading web interface HTML: {e}")
        return "<html><body><h1>Error loading interface</h1></body></html>"

def get_inline_ui_html():
    """Returns the complete web interface HTML with all assets embedded in a single call."""
    try:
        # Get the current directory path
        current_dir = pathlib.Path(__file__).parent
        
        # Read HTML file
        with open(current_dir / 'live_transcription.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Read CSS file        
        with open(current_dir / 'live_transcription.css', 'r', encoding='utf-8') as f:
            css_content = f.read()
            
        # Read JS file
        with open(current_dir / 'live_transcription.js', 'r', encoding='utf-8') as f:
            js_content = f.read()
        
        # SVG files
        with open(current_dir / 'src' / 'system_mode.svg', 'r', encoding='utf-8') as f:
            system_svg = f.read()
            system_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(system_svg.encode('utf-8')).decode('utf-8')}"
            
        with open(current_dir / 'src' / 'light_mode.svg', 'r', encoding='utf-8') as f:
            light_svg = f.read()
            light_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(light_svg.encode('utf-8')).decode('utf-8')}"
            
        with open(current_dir / 'src' / 'dark_mode.svg', 'r', encoding='utf-8') as f:
            dark_svg = f.read()
            dark_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(dark_svg.encode('utf-8')).decode('utf-8')}"
        
        # Replace external references
        html_content = html_content.replace(
            '<link rel="stylesheet" href="/web/live_transcription.css" />',
            f'<style>\n{css_content}\n</style>'
        )
        
        html_content = html_content.replace(
            '<script src="/web/live_transcription.js"></script>',
            f'<script>\n{js_content}\n</script>'
        )
        
        # Replace SVG references
        html_content = html_content.replace(
            '<img src="/web/src/system_mode.svg" alt="" />',
            f'<img src="{system_data_uri}" alt="" />'
        )
        
        html_content = html_content.replace(
            '<img src="/web/src/light_mode.svg" alt="" />',
            f'<img src="{light_data_uri}" alt="" />'
        )
        
        html_content = html_content.replace(
            '<img src="/web/src/dark_mode.svg" alt="" />',
            f'<img src="{dark_data_uri}" alt="" />'
        )
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error creating embedded web interface: {e}")
        return "<html><body><h1>Error loading embedded interface</h1></body></html>"


if __name__ == '__main__':
    
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
    from starlette.staticfiles import StaticFiles
    
    app = FastAPI()    
    web_dir = pathlib.Path(__file__).parent
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    
    @app.get("/")
    async def get():
        return HTMLResponse(get_inline_ui_html())

    uvicorn.run(app=app, host="0.0.0.0", port=8000)
