"use client"
import {Spectra} from "@/Spectra/handle";

export default function  DefaultSpectra( {setSpectrum, setDisplayedSpectrum} ) {
    const getdefault = async () => {
        try {
            const res = await fetch(
                `http://localhost:3000/api/defaultSpectrum`,{
                    method: 'GET',
                }
            );
            const data = await res.json();
            console.log(setSpectrum)
            let new_spectrum = new Spectra(data.x, data.y)
            setSpectrum(new_spectrum)
            setDisplayedSpectrum(new_spectrum)
        } catch (err) {
            console.log(err);
        }
    };


    return (
        <div className="w-full items-center justify-center border-black">
            <label
                className="flex items-center justify-center border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <p className="mb-2 text-sm text-gray-500 dark:text-gray-400 p-8 text-center"><span className="text-center font-semibold items-center justify-center">Load example Spectrum</span>
                    </p>
                </div>
                <input id="load_default_spectra" type="button" className="hidden" onClick={() => {
                    let data = getdefault();
                }}/>
            </label>
        </div>
    )
}