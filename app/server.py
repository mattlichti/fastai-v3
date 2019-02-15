from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1-gHpjXQmARxsKiax-mcpU1hssqwOm2a3'
export_file_name = 'model.pkl'

classes = ['Discmania_P2',
 'Discraft_Buzzz',
 'Dynamic Discs_Escape',
 'Dynamic Discs_Judge',
 'Dynamic Discs_Truth',
 'Gateway_Wizard',
 'Innova_Aviar',
 'Innova_Beast',
 'Innova_Boss',
 'Innova_Destroyer',
 'Innova_Firebird',
 'Innova_Katana',
 'Innova_Leopard',
 'Innova_Mako3',
 'Innova_Roc',
 'Innova_Roc3',
 'Innova_Shark',
 'Innova_Shryke',
 'Innova_Sidewinder',
 'Innova_Teebird',
 'Innova_Tern',
 'Innova_Thunderbird',
 'Innova_Valkyrie',
 'Innova_Wraith',
 'Westside_Harp']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    # return JSONResponse({'result': str(prediction)})

    pred_class,pred_idx,outputs = learn.predict(img)
    output = str(pred_class) + '<br> <br>Probabilities: <br>' 
    for idx, disc in enumerate(outputs):
        output += str(classes[idx]) + ': '
        output += str(round(disc.item()*100,1)) + '%' + '<br>'
    return JSONResponse({'result': output})




if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
