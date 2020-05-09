const Matrix = require('./Matrix');
const RedeNeural = require('./RedeNeural');

var nn = new RedeNeural(2, 3, 5);
var arr = [2, 3];
nn.feedfoward(arr);
//Funcionamento da NN feito
//Ensinando a aprender