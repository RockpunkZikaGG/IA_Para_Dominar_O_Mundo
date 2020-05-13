const Matrix = require('./Matrix');

function sigmoid(x){
    return 1/(1 + Math.exp(-x));
}
function d_sigmoid(x){
    return x * (1 - x);
}

module.exports = class RedeNeural{
    constructor(i_nodes, h_nodes, o_nodes){
        this.i_nodes = i_nodes;
        this.h_nodes = h_nodes;
        this.o_nodes = o_nodes;
        
        this.bias_ih = new Matrix(this.h_nodes, 1);
        this.bias_ih.randomize();

        this.bias_ho = new Matrix(this.o_nodes, 1);
        this.bias_ho.randomize();
        
        this.weigths_ih = new Matrix(this.h_nodes, this.i_nodes);
        this.weigths_ih.randomize();

        this.weigths_ho = new Matrix(this.o_nodes, this.h_nodes);
        this.weigths_ho.randomize();

        this.learning_rate = 0.2;
    }
    predict(arr){
        let input =  Matrix.arrayToMatrix(arr);

        let hidden = Matrix.multiply(this.weigths_ih, input);
        hidden = Matrix.add(hidden, this.bias_ih);
        hidden.map(sigmoid);
        
        let output = Matrix.multiply(this.weigths_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);
        
        output = Matrix.matrixToArray(output);
        return output;
    }
    train(arr, target){
        let input =  Matrix.arrayToMatrix(arr);
        
        //INPUT -> HIDDEN
        let hidden = Matrix.multiply(this.weigths_ih, input);
        hidden = Matrix.add(hidden, this.bias_ih);
        hidden.map(sigmoid);
        
        //HIDDEN -> OUTPUT
        let output = Matrix.multiply(this.weigths_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);

        //BACKPROPAGATION
        //OUTPUT -> HIDDEN 
        let expected = Matrix.arrayToMatrix(target);
        let output_erro = Matrix.subtract(expected, output);
        let d_output = Matrix.map(output, d_sigmoid);

        let hidden_T = Matrix.transpose(hidden);

        let gradient_O = Matrix.hadamard(d_output, output_erro);
        gradient_O = Matrix.escalar_multiply(gradient_O, this.learning_rate);

        this.bias_ho = Matrix.add(this.bias_ho, gradient_O);

        let weigths_ho_deltas = Matrix.multiply(gradient_O, hidden_T);
        this.weigths_ho = Matrix.add(this.weigths_ho, weigths_ho_deltas);
        
        //HIDDEN -> INPUT
        let weigths_ho_T = Matrix.transpose(this.weigths_ho);
        let hidden_erro = Matrix.multiply(weigths_ho_T, output_erro);
        let d_hidden = Matrix.map(hidden, d_sigmoid);
        let input_T = Matrix.transpose(input);

        let gradient_H = Matrix.hadamard(d_hidden, hidden_erro);
        gradient_H = Matrix.escalar_multiply(gradient_H, this.learning_rate)

        this.bias_ih = Matrix.add(this.bias_ih, gradient_H);

        let weigths_ih_deltas = Matrix.multiply(gradient_H, input_T);
        this.weigths_ih = Matrix.add(this.weigths_ih, weigths_ih_deltas);
    }
}