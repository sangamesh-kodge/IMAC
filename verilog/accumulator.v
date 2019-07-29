
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
module accumulator(
    	output wire [13:0] out,
    	input  [4:0] in,
    	input  clk ,
	input  en , 
	input  reset
    );

	reg [13:0] outreg;

	assign out = outreg;
	

	initial 
		begin 
			outreg<=13'b0;
		end



	always @(posedge clk or posedge reset )
		begin

			if (reset == 1'b1)
				outreg<=0;
			else
				if (en == 1'b1)
					outreg <= outreg+in;
				else 
					outreg <= outreg;		
		end
endmodule
















