"use client"

import DefaultSpectra from "@/Spectra/client_component";
import {Line} from "react-chartjs-2";
import {Dispatch, SetStateAction, useState} from "react";
import LineChart from "@/Spectra/SpectrumChart";
import Chart from "chart.js/auto";
import { CategoryScale } from "chart.js";
import {Spectra} from "@/Spectra/handle";
import {Upload2} from "@/Spectra/upload_2";
import {NextUIProvider} from "@nextui-org/react";
import Results from "@/Spectra/sendPrintSpectra";
import InteractionButtons from "@/Spectra/requestButtons";


Chart.register(CategoryScale);


export default function Home() {
    const [spectrum, setSpectrum] = useState<Spectra|null>(null);
    const [rows, setRows] = useState([]);
    const [preProcSpectrum, setPreProcSpectrum] = useState<Spectra|null>(null);
    const [displayedSpectrum, setDisplayedSpectrum] = useState<Spectra|null>(null);


    return (
    <NextUIProvider>
      <main className="min-h-screen w-full items-center p-24 self-center bg-white">

          <div className="grid grid-rows-5 grid-cols-6 gap-4 justify-center content-center">

              <div className="col-span-5 row-span-3 p-16 self-center h-full">
                  <LineChart spectrum={displayedSpectrum}/>
              </div>
              <div className="row-span-3 col-span-1 items-center">
                  <DefaultSpectra setSpectrum={setSpectrum} setDisplayedSpectrum = {setDisplayedSpectrum}/>
                  <Upload2 setSpectrum={setSpectrum} setDisplayedSpectrum = {setDisplayedSpectrum}/>
                  <InteractionButtons spectrum={spectrum}
                                      preProcSpectrum= {preProcSpectrum}
                                      displayedSpectrum={displayedSpectrum}
                                      setPreProcSpectrum= {setPreProcSpectrum}
                                      setDisplayedSpectrum = {setDisplayedSpectrum}
                                      setRows = {setRows}
                  />

              </div>
              <div className="col-span-6 row-span-2 self-center content-center">
                  <Results rows={rows}/>
              </div>
              </div>
      </main>
    </NextUIProvider>
);
}
