import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from maldi_nn.spectrum import SpectrumObject, Normalizer, VarStabilizer, BaselineCorrecter, Smoother, Binner

from src.ae import ae
from src.ms_data import MSData, Resistances

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

available_model = {
    "ae_based": ae()
}

@app.get("/")
async def root():
    return {"message": "I am alive"}

@app.get("/availableModel")
async def availableModel():
    return {"message": "I am alive"}

@app.post("/resistance/")
async def resistance(ms_data: MSData) -> Resistances:
    result = available_model["ae_based"].predict(ms_data)

    return result

normalizer = Normalizer()
varStabilizer = VarStabilizer()
baselineRemover = BaselineCorrecter(method="SNIP")
smoother = Smoother()
binner = Binner(step=1)

@app.post("/preprocessMS/")
async def preprocess(ms_data: MSData, stabilize:bool, smooth:bool, baselineRemoval:bool, intensityCalibrating:bool, binning: bool) -> MSData:
    spectrum = SpectrumObject(
        mz = np.asarray(ms_data.xValues),
        intensity= np.asarray(ms_data.yValues),
    )

    if stabilize:
        spectrum = varStabilizer(spectrum)

    if smooth:
        spectrum = smoother(spectrum)

    if baselineRemoval:
        spectrum = baselineRemover(spectrum)

    if intensityCalibrating:
        spectrum = normalizer(spectrum)

    if binning:
        spectrum = binner(spectrum)

    result_ms = MSData(xValues=spectrum.mz, yValues=spectrum.intensity)

    return result_ms
