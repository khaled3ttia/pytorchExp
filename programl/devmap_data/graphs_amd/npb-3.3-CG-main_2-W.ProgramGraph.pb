

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %4
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %16
 i1B

	full_text	

i1 %8
0shl8B'
%
	full_text

%10 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%11 = ashr exact i64 %10, 32
%i648B

	full_text
	
i64 %10
^getelementptr8BK
I
	full_text<
:
8%12 = getelementptr inbounds double, double* %0, i64 %11
%i648B

	full_text
	
i64 %11
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %12, align 8, !tbaa !8
-double*8B

	full_text

double* %12
^getelementptr8BK
I
	full_text<
:
8%13 = getelementptr inbounds double, double* %1, i64 %11
%i648B

	full_text
	
i64 %11
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %13, align 8, !tbaa !8
-double*8B

	full_text

double* %13
^getelementptr8BK
I
	full_text<
:
8%14 = getelementptr inbounds double, double* %2, i64 %11
%i648B

	full_text
	
i64 %11
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %14, align 8, !tbaa !8
-double*8B

	full_text

double* %14
^getelementptr8BK
I
	full_text<
:
8%15 = getelementptr inbounds double, double* %3, i64 %11
%i648B

	full_text
	
i64 %11
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %15, align 8, !tbaa !8
-double*8B

	full_text

double* %15
'br8B

	full_text

br label %16
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %0
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 0.000000e+00
#i328B

	full_text	

i32 0      	  
 

                     ! "     	 
  
  
  
       ## ## $ $ 
% % % % & "
main_2"
_Z13get_global_idj*?
npb-CG-main_2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?
 
transfer_bytes_log1p
??A

devmap_label
 

transfer_bytes
???

wgsize_log1p
??A