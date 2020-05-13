const Matrix = require('./Matrix');
const RedeNeural = require('./RedeNeural');

var nn = new RedeNeural(2, 3, 1);
var training = true;
var cont = 0;

/*nn.bias_ih.data = [
  [ -1.7554206719140355 ],
  [ -3.096357245915699 ],
  [ -2.8432889446395864 ]
];

nn.bias_ho.data = [ [ -3.4258535342008276 ] ];

nn.weigths_ih.data = [
  [ 3.827301169303692, 3.781412143398113 ],
  [ 6.393964968656268, 6.4350457122186375 ],
  [ 5.8937265674483275, 5.934204493333632 ]
];

nn.weigths_ho.data = [ [ 1.175124084415226, 3.4572399845714195, 2.8780381929383902 ] ];
*/
var teste = {
    input: [[0,0],[1,0],[0,1],[1,1]],
    output: [[0],[1],[1],[0]]
}

/*let index = 0;
console.log(Math.round(nn.predict(teste.input[index])));*/

while (training){
    cont++;
    var index = Math.floor(Math.random()*4);

    console.log(nn.predict(teste.input[index])[0]+" "+index+" "+cont);
    nn.train(teste.input[index], teste.output[index]);

    if( nn.predict(teste.input[0])[0] < 0.05 && nn.predict(teste.input[1])[0] > 0.95){
      training = false;
      console.log('tentativa: '+cont);
      console.log('bias_ih: ');
      console.log(nn.bias_ih.data);
      console.log('bias_ho: ');
      console.log(nn.bias_ho.data);
      console.log('weigths_ih: ');
      console.log(nn.weigths_ih.data);
      console.log('weigths_ho: ');
      console.log(nn.weigths_ho.data);
    }
    if(cont == 20000){
      training = false;
      console.log("Número de tantativas excedidas");
    }
}

console.log('Terminou');