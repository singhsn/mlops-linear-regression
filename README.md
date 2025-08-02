# mlops-linear-regression


1. **Clone the repository** 
   # repo url - https://github.com/singhsn/mlops-linear-regression


git clone git@github.com:singhsn/mlops-linear-regression.git
cd mlops-linear-regression
add all the required files 
git add .
git commit -m "mlops linear regression model "
git push origin main 



# quantization details 

Testing quantized model...
Original Model - R² Score: 0.5758, Loss: 0.5559
Quantized Model - R² Score: 0.5764, Loss: 0.5551
R² Score Difference: 0.000628
Model size reduction: 72 bytes -> 9 bytes
Compression ratio: 8.00x


Metric 	        Original Model 	    Quantized model (8 bit uint)	Difference 
R² Score	        0.5758	                    0.5764	                0.0006
MSE	                0.5559	                    0.5551	                0.0008
Model Size	        72 bytes	                9 bytes	                8x smaller
Compression Ratio	1x	                        8x	                    8.00x
			


