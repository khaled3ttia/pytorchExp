

[external]
KcallBC
A
	full_text4
2
0%3 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%4 = trunc i64 %3 to i32
"i64B

	full_text


i64 %3
3icmpB+
)
	full_text

%5 = icmp slt i32 %4, %1
"i32B

	full_text


i32 %4
6brB0
.
	full_text!

br i1 %5, label %6, label %10
 i1B

	full_text	

i1 %5
/shl8B&
$
	full_text

%7 = shl i64 %3, 32
$i648B

	full_text


i64 %3
7ashr8B-
+
	full_text

%8 = ashr exact i64 %7, 32
$i648B

	full_text


i64 %7
\getelementptr8BI
G
	full_text:
8
6%9 = getelementptr inbounds double, double* %0, i64 %8
$i648B

	full_text


i64 %8
Vstore8BK
I
	full_text<
:
8store double 0.000000e+00, double* %9, align 8, !tbaa !7
,double*8B

	full_text


double* %9
'br8B

	full_text

br label %10
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %1
-; undefined function B

	full_text

 
4double8B&
$
	full_text

double 0.000000e+00
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32      	  
 

           	 
             
"

init_mem_1"
_Z13get_global_idj*?
npb-CG-init_mem_1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
PѝA

devmap_label

 
transfer_bytes_log1p
PѝA

wgsize
?

transfer_bytes	
????