"use client"

import React from "react";
import { Line } from "react-chartjs-2";
import {Spectra} from "@/Spectra/handle";

interface InputLineChart {
    spectrum: Spectra|null
}
function LineChart( {spectrum} : InputLineChart) {


    let new_data = []

    if (spectrum != null){
        for (let i = 0; i < spectrum.x.length; ++i) {
            new_data[i] = {x:spectrum.x[i], y: spectrum.y[i]}
        }
    }

    return (
            <Line
                data={{
                    datasets: [
                        {
                            data: new_data
                        }
                    ]
                }}
                options={{
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            min: 2000,
                            max: 20000,
                        },
                        y: {
                            min: 0,
                        }
                    },
                    parsing: false,
                    plugins: {
                        title: {
                            display: true,
                            text: "Mass Spectra"
                        },
                        legend: {
                            display: false
                        },
                        decimation: {
                            enabled: true,
                            algorithm: 'lttb',
                            samples: 5000,
                            threshold: 5000,
                        }
                    }
                }}
            />
    );
}
export default LineChart;