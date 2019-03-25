from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://www.dropbox.com/s/c7k3z8rcq33buxy/75discs.pkl?dl=1'
export_file_name = '75discs.pkl'


classes = ['Axiom Envy',
 'Discmania FD',
 'Discmania P2',
 'Discraft Avenger SS',
 'Discraft Buzzz',
 'Discraft Heat',
 'Discraft Nuke',
 'Discraft Undertaker',
 'Discraft Zone',
 'Dynamic Discs Convict',
 'Dynamic Discs Defender',
 'Dynamic Discs Deputy',
 'Dynamic Discs Emac Truth',
 'Dynamic Discs Enforcer',
 'Dynamic Discs Escape',
 'Dynamic Discs Felon',
 'Dynamic Discs Freedom',
 'Dynamic Discs Judge',
 'Dynamic Discs Justice',
 'Dynamic Discs Maverick',
 'Dynamic Discs Renegade',
 'Dynamic Discs Sheriff',
 'Dynamic Discs Trespass',
 'Dynamic Discs Truth',
 'Dynamic Discs Verdict',
 'Dynamic Discs Warden',
 'Dynamic Discs Witness',
 'Gateway Wizard',
 'Innova Archangel',
 'Innova Atlas',
 'Innova Aviar',
 'Innova Aviar3',
 'Innova Beast',
 'Innova Boss',
 'Innova Colossus',
 'Innova Colt',
 'Innova Daedalus',
 'Innova Dart',
 'Innova Destroyer',
 'Innova Dragon',
 'Innova Eagle',
 'Innova Firebird',
 'Innova Gator',
 'Innova Katana',
 'Innova Leopard',
 'Innova Leopard3',
 'Innova Mako3',
 'Innova Mamba',
 'Innova Nova',
 'Innova Orc',
 'Innova Rhyno',
 'Innova Roadrunner',
 'Innova Roc',
 'Innova Roc3',
 'Innova Shark',
 'Innova Shryke',
 'Innova Sidewinder',
 'Innova Stingray',
 'Innova TL',
 'Innova Teebird',
 'Innova Tern',
 'Innova Thunderbird',
 'Innova Valkyrie',
 'Innova Vulcan',
 'Innova Wraith',
 'Latitude 64 Ballista',
 'Latitude 64 Compass',
 'Latitude 64 Dagger',
 'Latitude 64 Explorer',
 'Latitude 64 Pure',
 'Latitude 64 River',
 'Latitude 64 Saint',
 'MVP Tesla',
 'MVP Volt',
 'Westside Harp']

pclasses = ['Discmania C-line',
 'Discmania D-line',
 'Discmania S-line',
 'Discraft Big-z',
 'Discraft Elite X',
 'Discraft Elite Z',
 'Discraft Esp',
 'Discraft Pro-d',
 'Discraft Titanium',
 'Dynamic Discs Biofuzion',
 'Dynamic Discs Classic',
 'Dynamic Discs Classic-blend',
 'Dynamic Discs Fuzion',
 'Dynamic Discs Lucid',
 'Dynamic Discs Moonshine',
 'Dynamic Discs Prime',
 'Innova Blizzard',
 'Innova Champion',
 'Innova DX',
 'Innova Glow-champion',
 'Innova Gstar',
 'Innova Kc-pro',
 'Innova Pro',
 'Innova R-pro',
 'Innova Star',
 'Innova Starlite',
 'Innova Xt',
 'Latitude 64 Gold Line',
 'Latitude 64 Opto Line',
 'Latitude 64 Zero-hard',
 'MVP Neutron',
 'Westside Bt-hard',
 'Westside Origio',
 'Westside Tournament',
 'Westside Vip']

plastics_export_file_url = 'https://www.dropbox.com/s/8qtxued68q92may/35plastics.pkl?dl=1'
plastics_export_file_name = '35plastics.pkl'

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
    await download_file(plastics_export_file_url, path/plastics_export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        plastics_learn = load_learner(path, plastics_export_file_name)
        return learn, plastics_learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

# async def setup_plastic_learner():
#     await download_file(plastics_export_file_url, path/plastics_export_file_name)
#     try:
#         plastics_learn = load_learner(path, plastics_export_file_name)
#         return plastics_learn
#     except RuntimeError as e:
#         if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
#             print(e)
#             message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
#             raise RuntimeError(message)
#         else:
#             raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn, plastic_learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
# plastic_tasks = [asyncio.ensure_future(setup_plastic_learner())]
# plastic_learn = loop.run_until_complete(asyncio.gather(*plastic_tasks))[0]

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

    pred_class,pred_idx,outputs = learn.predict(img)

    output = str(pred_class) + '<br> <br>Probabilities: <br>' 
    for idx in np.argsort(-outputs):
        output += str(classes[idx]) + ': '
        output += str(round(outputs[idx].item()*100,1)) + '%' + '<br>'
        
    return JSONResponse({'result': output})




if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
