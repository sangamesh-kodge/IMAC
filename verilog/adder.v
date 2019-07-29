

//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    13:56:58 07/10/2019 
// Design Name: 
// Module Name:    Accumulator
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module adder(
    	output wire [5:0] out,
    	input  [4:0] in1,
    	input  [4:0] in2
    );

	
	assign out = in1 +in2;
	
endmodule















