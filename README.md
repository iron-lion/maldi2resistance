# Toward a foundation model for antimicrobial resistance using mass spectrometry data

The aim of this project is to investigate whether individual models trained on the DRIAMS 
dataset are able to predict antibiotic resistance. This problem is modeled as a multi-class multi-label problem. 

The aim of this project is to investigate whether individual models trained on the DRIAMS dataset are able to predict 
antibiotic resistance. This problem is modeled as a multi-class multi-label problem. 

Furthermore, this repository contains the python package maldi2resistance. The main functionality of this package is to 
provide the DRIAMS dataset in a convenient way for training neural networks in the context of pytorch. Thus, 
antibiotics, location and year can be defined, but also other parameters to fine-tune the loaded spectra to 
individual needs. Furthermore, it is possible to load these spectra into the cache to enable the fastest possible 
training. Since the preprocessing built into this reduces the DRIAMS dataset to less than 8GB, this is also 
realistic on 'everyday' computers.

## Repository Structure

    ./
    ├── ablation
    │   ├── removeOver10000
    │   ├── removeUnder10000
    │   └── window2000
    ├── additional_model
    │   ├── AE
    │   ├── MaskedLossWeight
    │   ├── MSDeepAMR
    │   └── ResMLP
    ├── data 
    ├── backend
    ├── dataset_performance
    ├── frontend
    ├── IntegratedGradiens
    ├── preTraining
    │   ├── trainOnSiteA
    │   └── trainOnSiteB
    └── src
        └── maldi2resistance

## Front & Backend
The front and backend allow you to upload an unprocessed MALDI-TOF spectrum, preprocess it, display both 
versions of the spectrum graphically and then have a model predict the resistances based on the spectrum.

A live [example can be found here.](http://89.58.62.37:3000/)

### Deployment

There are several ways to deploy the frontend and backend. This article briefly explains how this was solved using 
systemd and defining a service for nextjs to run the frontend and a service running fastapi/uvicorn to serve the backend.

> **_NOTE:_**  Please be aware that it is also necessary to adjust the links within the frontend that refer to 
> itself and to the backend and to change them to the correct IP/domain.

<details>
<summary>Example service file for nextjs</summary>
<br>

    [Unit]
    Description=NextJS-Server
    After=network-online.target
    
    [Service]
    
    User=<user>
    WorkingDirectory=<directory>
    ExecStart=/usr/bin/npm run dev
    
    [Install]
    WantedBy=multi-user.target

</details>

<details>
<summary>Example service file for fastapi/uvicorn</summary>
<br>

    [Unit]
    Description=FastAPI-Server
    After=network-online.target
    
    [Service]
    
    User=<user>
    WorkingDirectory=<directory>
    ExecStart=<PythonPath> -m uvicorn src.server:app --host 0.0.0.0 --port 8000
    
    [Install]
    WantedBy=multi-user.target
</details>