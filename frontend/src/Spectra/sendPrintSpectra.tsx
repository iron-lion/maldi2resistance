import {getKeyValue, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow} from "@nextui-org/react";
import {Dispatch, SetStateAction, useEffect, useState} from "react";
import {number} from "prop-types";
import {Spectra} from "@/Spectra/handle";

interface InputResults {
    rows: never[]
}

export default function  Results( {rows}:InputResults ) {


    let columns = [
        {
            key: "antibiotic",
            label: "Antibiotic",
        },
        {
            key: "resistant",
            label: "Resistant",
        },
    ];



    return (
        <Table isStriped
               aria-label="Example table with dynamic content"
               className="dark:bg-slate-800 border-black"
        >
            <TableHeader columns={columns}>
                {(column) => <TableColumn key={column.key}>{column.label}</TableColumn>}
            </TableHeader>
            <TableBody items={rows}>
                {(item) => (
                    <TableRow key={item.key}>
                        {(columnKey) => <TableCell>{getKeyValue(item, columnKey)}</TableCell>}
                    </TableRow>
                )}
            </TableBody>
        </Table>
    )
}