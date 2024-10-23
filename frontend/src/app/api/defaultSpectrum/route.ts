import { promises as fs } from 'fs';
import {Spectra} from "@/Spectra/handle";
import {NextRequest, NextResponse} from "next/server";

export async function GET(request: NextRequest) {
    const file = await fs.readFile(process.cwd() + '/src/data/ExampleSpectra_old.txt', 'utf8')

    let spectra = Spectra.from_string(file);

    return NextResponse.json(spectra)
}