"use client"

import React, {Dispatch, SetStateAction, useEffect, useState} from "react";
import {Button, Switch} from "@nextui-org/react";
import {Spacer} from "@nextui-org/spacer";
import {Spectra} from "@/Spectra/handle";
import {number} from "prop-types";

interface InputInteractionButtons {
    spectrum: Spectra | null
    preProcSpectrum : Spectra | null
    displayedSpectrum: Spectra | null
    setPreProcSpectrum: Dispatch<SetStateAction<Spectra | null>>
    setDisplayedSpectrum: Dispatch<SetStateAction<Spectra | null>>
    setRows:Dispatch<SetStateAction<never[]>>
}
export default function  InteractionButtons( {spectrum, preProcSpectrum, displayedSpectrum, setPreProcSpectrum, setDisplayedSpectrum, setRows} :InputInteractionButtons ) {
    const [switchDisabled, setSwitchDisabled] = useState(true);
    const [switchSelected, setSwitchSelected] = useState(false);

    async function getPreProc() {

        const res = await fetch('http://127.0.0.1:8000/preprocessMS/?stabilize=true&smooth=true&baselineRemoval=true&intensityCalibrating=true&binning=true',{
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                xValues: spectrum.x,
                yValues: spectrum.y,
            }),
        })

        if (!res.ok) {
            // This will activate the closest `error.js` Error Boundary
            throw new Error('Failed to fetch data')
        }

        const content = await res.json();
        let newPreProcSpectrum = new Spectra(content.xValues,content.yValues)
        setPreProcSpectrum(newPreProcSpectrum)
        setDisplayedSpectrum(newPreProcSpectrum)
        setSwitchDisabled(false)
        setSwitchSelected(true)
    }


    function swapDisplayed(event) {
        setSwitchSelected(!switchSelected)

        if (event.target.checked){
            setDisplayedSpectrum(preProcSpectrum)
        } else {
            setDisplayedSpectrum(spectrum)
        }

    }

    async function getResistance() {
        setRows([])

        const res = await fetch('http://127.0.0.1:8000/resistance/',{
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                xValues: displayedSpectrum.x,
                yValues: displayedSpectrum.y,
            }),
        })

        if (!res.ok) {
            // This will activate the closest `error.js` Error Boundary
            throw new Error('Failed to fetch data')
        }

        const content = await res.json();

        let results = content.resistances
        let row_buffer = []

        let n: number = 0
        results.forEach(function (value) {
            row_buffer.push(
                {
                    "key": n,
                    "antibiotic":value.antibioticName,
                    "resistant":value.antibioticResistance,
                }
            )
            n = n+1;
        })
        setRows(row_buffer)

    }


    return (
        <div className = "grid grid-rows-3 grid-cols-1 p-10 items-center gap-8 m-4">
            <Switch isSelected={switchSelected} isDisabled = {switchDisabled} onChange={swapDisplayed}>
                Preprocessed
            </Switch>
            <Spacer x={4} />
            <Button color="primary" size="lg" onClick={getPreProc}>
                Preprocess MS data
            </Button>
            <Spacer x={4} />
            <Button color="primary" size="lg" onClick={getResistance}>
                Run Model
            </Button>
        </div>
    )
}