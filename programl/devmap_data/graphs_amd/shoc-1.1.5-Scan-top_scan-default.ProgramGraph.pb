

[external]
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_local_idj(i32 0) #3
2sextB*
(
	full_text

%5 = sext i32 %1 to i64
3icmpB+
)
	full_text

%6 = icmp ult i64 %4, %5
"i64B

	full_text


i64 %4
"i64B

	full_text


i64 %5
5brB/
-
	full_text 

br i1 %6, label %9, label %7
 i1B

	full_text	

i1 %6
hcall8B^
\
	full_textO
M
K%8 = tail call float @scanLocalMem(float 0.000000e+00, float* %2, i32 1) #4
'br8B

	full_text

br label %13
[getelementptr8BH
F
	full_text9
7
5%10 = getelementptr inbounds float, float* %0, i64 %4
$i648B

	full_text


i64 %4
Lload8BB
@
	full_text3
1
/%11 = load float, float* %10, align 4, !tbaa !8
+float*8B

	full_text


float* %10
`call8BV
T
	full_textG
E
C%12 = tail call float @scanLocalMem(float %11, float* %2, i32 1) #4
)float8B

	full_text

	float %11
Lstore8BA
?
	full_text2
0
.store float %12, float* %10, align 4, !tbaa !8
)float8B

	full_text

	float %12
+float*8B

	full_text


float* %10
'br8B

	full_text

br label %13
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %1
*float*8B

	full_text

	float* %0
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
2float8B%
#
	full_text

float 0.000000e+00       	 

            
    
   
  
  	              "

top_scan"
_Z12get_local_idj"
scanLocalMem*?
shoc-1.1.5-Scan-top_scan.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize
?

transfer_bytes
???

wgsize_log1p
cA
 
transfer_bytes_log1p
cA

devmap_label
