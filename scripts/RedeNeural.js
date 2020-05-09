const Matrix = require('./Matrix');

function sigmoid(x){
    return 1/(1 + Math.exp(-x));
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
    }
    feedfoward(in_data){
        let input =  Matrix.arrayToMatrix(in_data);

        let hidden = Matrix.multiply(this.weigths_ih, input);
        hidden = Matrix.add(hidden, this.bias_ih);
        hidden.map(sigmoid);
        
        let output = Matrix.multiply(this.weigths_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);
        
        output.print();
    }
}