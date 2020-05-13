const Matrix = require('./Matrix');

function sigmoid(x){
    return 1/(1 + Math.exp(-x));
}
function d_sigmoid(x){
    return x * (1 - x);
}

module.exports = class RedeNeural{
    constructor(i_nodes, h_nodes_1, h_nodes_2, o_nodes){
        this.i_nodes = i_nodes;
        this.h_nodes_1 = h_nodes_1;
        this.h_nodes_2 = h_nodes_2;
        this.o_nodes = o_nodes;
        
        this.bias_ih = new Matrix(this.h_nodes_1, 1);
        this.bias_ih.randomize();

        this.bias_hh = new Matrix(this.h_nodes_2, 1);
        this.bias_hh.randomize();

        this.bias_ho = new Matrix(this.o_nodes, 1);
        this.bias_ho.randomize();
        
        this.weigths_ih = new Matrix(this.h_nodes_1, this.i_nodes);
        this.weigths_ih.randomize();

        this.weigths_hh = new Matrix(this.h_nodes_2, this.h_nodes_1);
        this.weigths_hh.randomize();

        this.weigths_ho = new Matrix(this.o_nodes, this.h_nodes_2);
        this.weigths_ho.randomize();

        this.learning_rate = 0.2;
    }
    predict(arr){
        let input =  Matrix.arrayToMatrix(arr);

        let hidden_1 = Matrix.multiply(this.weigths_ih, input);
        hidden_1 = Matrix.add(hidden_1, this.bias_ih);
        hidden_1.map(sigmoid);

        let hidden_2 = Matrix.multiply(this.weigths_hh, hidden_1);
        hidden_2 = Matrix.add(hidden_2, this.bias_hh);
        hidden_2.map(sigmoid);
        
        let output = Matrix.multiply(this.weigths_ho, hidden_2);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);
        
        output = Matrix.matrixToArray(output);
        return output;
    }
    train(arr, target){
        let input =  Matrix.arrayToMatrix(arr);
        
        //INPUT -> HIDDEN
        let hidden_1 = Matrix.multiply(this.weigths_ih, input);
        hidden_1 = Matrix.add(hidden_1, this.bias_ih);
        hidden_1.map(sigmoid);

        //HIDDEN -> HIDDEN
        let hidden_2 = Matrix.multiply(this.weigths_hh, hidden_1);
        hidden_2 = Matrix.add(hidden_2, this.bias_hh);
        hidden_2.map(sigmoid);
        
        //HIDDEN -> OUTPUT
        let output = Matrix.multiply(this.weigths_ho, hidden_2);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);

        //BACKPROPAGATION
        //OUTPUT -> HIDDEN 
        let expected = Matrix.arrayToMatrix(target);
        let output_erro = Matrix.subtract(expected, output);
        let d_output = Matrix.map(output, d_sigmoid);
        let hidden_2_T = Matrix.transpose(hidden_2);

        let gradient_O = Matrix.hadamard(d_output, output_erro);
        gradient_O = Matrix.escalar_multiply(gradient_O, this.learning_rate);

        this.bias_ho = Matrix.add(this.bias_ho, gradient_O);

        let weigths_ho_deltas = Matrix.multiply(gradient_O, hidden_2_T);
        this.weigths_ho = Matrix.add(this.weigths_ho, weigths_ho_deltas);
        
        //HIDDEN -> HIDDEN
        let weigths_ho_T = Matrix.transpose(this.weigths_ho);
        let hidden_2_erro = Matrix.multiply(weigths_ho_T, output_erro);
        let d_hidden_2 = Matrix.map(hidden_2, d_sigmoid);
        let hidden_1_T = Matrix.transpose(hidden_1);

        let gradient_H_2 = Matrix.hadamard(d_hidden_2, hidden_2_erro);
        gradient_H_2 = Matrix.escalar_multiply(gradient_H_2, this.learning_rate)

        this.bias_hh = Matrix.add(this.bias_hh, gradient_H_2);

        let weigths_hh_deltas = Matrix.multiply(gradient_H_2, hidden_1_T);
        this.weigths_hh = Matrix.add(this.weigths_hh, weigths_hh_deltas);

        //HIDDEN -> INPUT
        let weigths_hh_T = Matrix.transpose(this.weigths_hh);
        let hidden_1_erro = Matrix.multiply(weigths_hh_T, hidden_2_erro);
        let d_hidden_1 = Matrix.map(hidden_1, d_sigmoid);
        let input_T = Matrix.transpose(input);

        let gradient_H_1 = Matrix.hadamard(d_hidden_1, hidden_1_erro);
        gradient_H_1 = Matrix.escalar_multiply(gradient_H_1, this.learning_rate)

        this.bias_ih = Matrix.add(this.bias_ih, gradient_H_1);

        let weigths_ih_deltas = Matrix.multiply(gradient_H_1, input_T);
        this.weigths_ih = Matrix.add(this.weigths_ih, weigths_ih_deltas);
    }
}