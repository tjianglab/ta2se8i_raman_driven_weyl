set style data dots
set nokey
set xrange [0: 3.18994]
set yrange [-11.40832 :  8.00164]
set arrow from  0.48508, -11.40832 to  0.48508,   8.00164 nohead
set arrow from  0.67975, -11.40832 to  0.67975,   8.00164 nohead
set arrow from  0.94098, -11.40832 to  0.94098,   8.00164 nohead
set arrow from  1.18649, -11.40832 to  1.18649,   8.00164 nohead
set arrow from  1.42606, -11.40832 to  1.42606,   8.00164 nohead
set arrow from  1.68275, -11.40832 to  1.68275,   8.00164 nohead
set arrow from  1.87879, -11.40832 to  1.87879,   8.00164 nohead
set arrow from  2.33468, -11.40832 to  2.33468,   8.00164 nohead
set arrow from  2.78741, -11.40832 to  2.78741,   8.00164 nohead
set xtics (" G "  0.00000," T "  0.48508,"D/B"  0.67975," Z "  0.94098,"A/C"  1.18649," Y "  1.42606,"H/G"  1.68275,"T/Y"  1.87879," G "  2.33468,"Z/G"  2.78741," L "  3.18994)
 plot "wannier90_band.dat"
