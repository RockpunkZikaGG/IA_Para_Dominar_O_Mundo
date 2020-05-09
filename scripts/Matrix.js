module.exports = class Matrix{
    constructor(rows, cols){
        this.rows = rows;
        this.cols = cols;

        this.data = [];
        for(let i=0; i<rows; i++){
            let arr = [];
            for(let j=0; j<cols; j++){
                arr.push(0);
            }
            this.data.push(arr);
        }
    }
    print(){
        console.table(this.data);
    }
    randomize(){
        this.map((ele, i, j) => {
            return Math.random()*2 - 1;
        })
    }
    map(func){
        return this.data = this.data.map((arr, i) => {
            return arr.map((num, j) => {
                return func(num, i, j);
            })
        });
    }
    static map(A, func){
        let matrix = new Matrix(A.rows, B.rows);
        matrix.data = this.data.map((arr, i) => {
            return arr.map((num, j) => {
                return func(num, i, j);
            })
        });
        return matrix;
    }
    static arrayToMatrix(arr){
        let matrix = new Matrix(arr.length, 1);
        matrix.map((elm, i, j) => {
            return arr[i];
        })
        return matrix;
    }
    static add(A, B){
        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((num, i, j) => {
            return A.data[i][j] + B.data[i][j];
        });
        return matrix;
    }
    static multiply(A, B){
        var matrix = new Matrix(A.rows, B.cols);
        matrix.map((num, i, j) => {
            let sum = 0;
            for(let k=0; k < A.cols; k++){
                sum += A.data[i][k] * B.data[k][j];
            }
            return sum;
        });
        return matrix;
    }
}