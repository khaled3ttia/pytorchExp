
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
5icmpB-
+
	full_text

%8 = icmp sgt i32 %7, 7000
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %21, label %9
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
8%14 = getelementptr inbounds double, double* %3, i64 %11
%i648B

	full_text
	
i64 %11
Abitcast8B4
2
	full_text%
#
!%15 = bitcast double* %14 to i64*
-double*8B

	full_text

double* %14
Hload8B>
<
	full_text/
-
+%16 = load i64, i64* %15, align 8, !tbaa !8
'i64*8B

	full_text


i64* %15
^getelementptr8BK
I
	full_text<
:
8%17 = getelementptr inbounds double, double* %2, i64 %11
%i648B

	full_text
	
i64 %11
Abitcast8B4
2
	full_text%
#
!%18 = bitcast double* %17 to i64*
-double*8B

	full_text

double* %17
Hstore8B=
;
	full_text.
,
*store i64 %16, i64* %18, align 8, !tbaa !8
%i648B

	full_text
	
i64 %16
'i64*8B

	full_text


i64* %18
^getelementptr8BK
I
	full_text<
:
8%19 = getelementptr inbounds double, double* %4, i64 %11
%i648B

	full_text
	
i64 %11
Abitcast8B4
2
	full_text%
#
!%20 = bitcast double* %19 to i64*
-double*8B

	full_text

double* %19
Hstore8B=
;
	full_text.
,
*store i64 %16, i64* %20, align 8, !tbaa !8
%i648B

	full_text
	
i64 %16
'i64*8B

	full_text


i64* %20
'br8B

	full_text

br label %21
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %3
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
&i328B

	full_text


i32 7000      	  
 

                       !" !! #$ ## %& %' %% (* + , !- .     	 
  
  
   
     
 "! $ &# ' ) ( ) ) // // 0 1 1 
2 2 3 "
conj_grad_0"
_Z13get_global_idj*?
npb-CG-conj_grad_0.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

devmap_label
 

wgsize
?
 
transfer_bytes_log1p
??A

wgsize_log1p
??A

transfer_bytes
???