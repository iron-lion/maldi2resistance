"use client";

import { FileInput, Label } from "flowbite-react";
import {Spectra} from "@/Spectra/handle";
import {Dispatch, SetStateAction} from "react";

interface InputUpload2 {
    setSpectrum: Dispatch<SetStateAction<Spectra | null>>
    setDisplayedSpectrum : Dispatch<SetStateAction<Spectra | null>>
}
export function Upload2({setSpectrum, setDisplayedSpectrum} :InputUpload2) {
    function setSpectra(e: any){

        if (!e.target.files) {
            return;
        }

        let spectrum_promised = Spectra.from_file(e.target.files[0])
        spectrum_promised.then( (spectra) => {
            console.log(spectra)
            setSpectrum(spectra)
            setDisplayedSpectrum(spectra)
        })

    }

    return (
        <div className="place-self-center items-center justify-center justify-items-center" >
            <Label
                htmlFor="dropzone-file"
                className="flex cursor-pointer items-center justify-center rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-700 dark:hover:border-gray-500 dark:hover:bg-gray-600"
            >
                <div className="flex flex-col items-center justify-center pb-6 pt-5">
                    <svg
                        className="mb-4 h-8 w-8 text-gray-500 dark:text-gray-400"
                        aria-hidden="true"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 20 16"
                    >
                        <path
                            stroke="currentColor"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
                        />
                    </svg>
                    <p className="mb-2 text-sm text-gray-500 dark:text-gray-400 text-center">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Mass spectrum</p>
                </div>
                <FileInput id="dropzone-file" className="hidden" onChange = {setSpectra} onDrop = {(event) => event.preventDefault()}/>
            </Label>
        </div>
    );
}