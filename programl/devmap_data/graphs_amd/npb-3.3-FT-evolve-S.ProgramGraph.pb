

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 2) #2
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #2
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #2
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
4icmpB,
*
	full_text

%13 = icmp slt i32 %8, %5
"i32B

	full_text


i32 %8
5icmpB-
+
	full_text

%14 = icmp slt i32 %10, %4
#i32B

	full_text
	
i32 %10
/andB(
&
	full_text

%15 = and i1 %13, %14
!i1B

	full_text


i1 %13
!i1B

	full_text


i1 %14
5icmpB-
+
	full_text

%16 = icmp slt i32 %12, %3
#i32B

	full_text
	
i32 %12
/andB(
&
	full_text

%17 = and i1 %15, %16
!i1B

	full_text


i1 %15
!i1B

	full_text


i1 %16
8brB2
0
	full_text#
!
br i1 %17, label %18, label %35
!i1B

	full_text


i1 %17
4mul8B+
)
	full_text

%19 = mul nsw i32 %8, %4
$i328B

	full_text


i32 %8
3add8B*
(
	full_text

%20 = add nsw i32 %3, 1
2add8B)
'
	full_text

%21 = add i32 %19, %10
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %10
2mul8B)
'
	full_text

%22 = mul i32 %21, %20
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %20
6add8B-
+
	full_text

%23 = add nsw i32 %22, %12
%i328B

	full_text
	
i32 %22
%i328B

	full_text
	
i32 %12
6sext8B,
*
	full_text

%24 = sext i32 %23 to i64
%i328B

	full_text
	
i32 %23
ygetelementptr8Bf
d
	full_textW
U
S%25 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %24, i32 0
%i648B

	full_text
	
i64 %24
Nload8BD
B
	full_text5
3
1%26 = load double, double* %25, align 8, !tbaa !8
-double*8B

	full_text

double* %25
^getelementptr8BK
I
	full_text<
:
8%27 = getelementptr inbounds double, double* %2, i64 %24
%i648B

	full_text
	
i64 %24
Oload8BE
C
	full_text6
4
2%28 = load double, double* %27, align 8, !tbaa !13
-double*8B

	full_text

double* %27
7fmul8B-
+
	full_text

%29 = fmul double %26, %28
+double8B

	full_text


double %26
+double8B

	full_text


double %28
ygetelementptr8Bf
d
	full_textW
U
S%30 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %24, i32 1
%i648B

	full_text
	
i64 %24
Oload8BE
C
	full_text6
4
2%31 = load double, double* %30, align 8, !tbaa !14
-double*8B

	full_text

double* %30
7fmul8B-
+
	full_text

%32 = fmul double %28, %31
+double8B

	full_text


double %28
+double8B

	full_text


double %31
Dstore8B9
7
	full_text*
(
&store double %29, double* %25, align 8
+double8B

	full_text


double %29
-double*8B

	full_text

double* %25
Dstore8B9
7
	full_text*
(
&store double %32, double* %30, align 8
+double8B

	full_text


double %32
-double*8B

	full_text

double* %30
ygetelementptr8Bf
d
	full_textW
U
S%33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %24, i32 0
%i648B

	full_text
	
i64 %24
Nstore8BC
A
	full_text4
2
0store double %29, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %29
-double*8B

	full_text

double* %33
ygetelementptr8Bf
d
	full_textW
U
S%34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %24, i32 1
%i648B

	full_text
	
i64 %24
Ostore8BD
B
	full_text5
3
1store double %32, double* %34, align 8, !tbaa !14
+double8B

	full_text


double %32
-double*8B

	full_text

double* %34
'br8B

	full_text

br label %35
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %2
6struct*8B'
%
	full_text

%struct.dcomplex* %1
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
6struct*8B'
%
	full_text

%struct.dcomplex* %0
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1       	  
 

                      !" !# !! $% $$ &' && () (( *+ ** ,- ,, ./ .0 .. 12 11 34 33 56 57 55 89 8: 88 ;< ;= ;; >? >> @A @B @@ CD CC EF EG EE HJ *K >K CL L M 
N N O &O 1   	  
             " #! %$ '& )$ +* -( /, 0$ 21 4, 63 7. 9& :5 <1 =$ ?. A> B$ D5 FC G  IH I I PP PP  PP  PP Q R R &R >S S S 1S C"
evolve"
_Z13get_global_idj*?
npb-FT-evolve.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize_log1p
@s?A

devmap_label

 
transfer_bytes_log1p
@s?A

transfer_bytes
???

wgsize
@