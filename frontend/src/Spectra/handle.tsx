class Spectra {
    x: number[];
    y: number[];

    constructor(x: number[], y:number[]) {
    this.x = x;
    this.y = y;
    }

    public static from_file(file) {
        console.log(file)

        var reader = new FileReader();
        return new Promise<Spectra>((resolve, reject) => {
            reader.addEventListener("loadend", function (event) {
                let text = event.target.result;
                resolve(Spectra.from_string(text));
            })
            reader.readAsText(file);
        });
    }

    public static from_string(string: string){
        var line = string.split(/\r\n|\n/);

        var x = [];
        var y = [];

        for (var i = 1; i < line.length; i++) {
            let currentLine = line[i]

            if (currentLine.charAt(0) == "#" || currentLine.charAt(0) == "\"" || !currentLine) {
                continue
            }

            var data = line[i].split(' ');
            x.push(Number(data[0]))
            y.push(Number(data[1]))
        }
        return new Spectra(x, y)
    }
}

function ms_csv2chart(file, canvas, spinner){
    var reader = new FileReader();
    reader.addEventListener("loadend", function(event) {

        let text = event.target.result;
        var line = text.split(/\r\n|\n/);

        var x = [];
        var y = [];

        for (var i=1; i<line.length; i++) {
            let currentLine = line[i]

            if (currentLine.charAt(0) == "#" || currentLine.charAt(0) == "\"" || !currentLine){
                continue
            }

            var data = line[i].split(' ');
            x.push(Number(data[0]))
            y.push(Number(data[1]))
        }

        const myChart = new Chart(canvas, {
            type: "line",
            data: {
                labels: x,
                datasets: [{
                    pointRadius: 2,
                    pointBackgroundColor: "rgba(0,0,255,1)",
                    data: y
                }]
            },
            options:{
                animation: false,
                spanGaps: true,
                showLine: false,
                scales: {
                    x: {
                        ticks: {
                            callback: function(value, index, ticks) {
                                label = this.getLabelForValue(value)
                                return Math.floor(label)
                            }
                        }
                    }
                }
            }
        });

        spinner.style.display="none";
        canvas.style.display ="block";
        send_ms_data({x: x, y:y})
    });
    reader.readAsText(file);
}

export { Spectra }