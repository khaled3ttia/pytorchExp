

[external]
KcallBC
A
	full_text4
2
0%2 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%3 = trunc i64 %2 to i32
"i64B

	full_text


i64 %2
6icmpB.
,
	full_text

%4 = icmp sgt i32 %3, 75000
"i32B

	full_text


i32 %3
5brB/
-
	full_text 

br i1 %4, label %9, label %5
 i1B

	full_text	

i1 %4
/shl8B&
$
	full_text

%6 = shl i64 %2, 32
$i648B

	full_text


i64 %2
7ashr8B-
+
	full_text

%7 = ashr exact i64 %6, 32
$i648B

	full_text


i64 %6
\getelementptr8BI
G
	full_text:
8
6%8 = getelementptr inbounds double, double* %0, i64 %7
$i648B

	full_text


i64 %7
Vstore8BK
I
	full_text<
:
8store double 1.000000e+00, double* %8, align 8, !tbaa !8
,double*8B

	full_text


double* %8
&br8B

	full_text

br label %9
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
'i328B

	full_text

	i32 75000
4double8B&
$
	full_text

double 1.000000e+00
$i648B

	full_text


i64 32      	  
 

          	 
              
"
main_1"
_Z13get_global_idj*?
npb-CG-main_1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
PѝA

wgsize_log1p
PѝA

devmap_label
 

wgsize
?

transfer_bytes	
????