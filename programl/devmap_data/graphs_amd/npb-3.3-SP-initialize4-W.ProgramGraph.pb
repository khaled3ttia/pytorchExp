

[external]
?allocaB5
3
	full_text&
$
"%6 = alloca [5 x double], align 16
BbitcastB7
5
	full_text(
&
$%7 = bitcast [5 x double]* %6 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %6
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %7) #4
"i8*B

	full_text


i8* %7
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #5
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #5
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %9, %4
"i32B

	full_text


i32 %9
5icmpB-
+
	full_text

%13 = icmp slt i32 %11, %2
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%14 = and i1 %12, %13
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %13
8brB2
0
	full_text#
!
br i1 %14, label %15, label %37
!i1B

	full_text


i1 %14
<sitofp8B0
.
	full_text!

%16 = sitofp i32 %9 to double
$i328B

	full_text


i32 %9
Ffmul8B<
:
	full_text-
+
)%17 = fmul double %16, 0x3F9D41D41D41D41D
+double8B

	full_text


double %16
=sitofp8B1
/
	full_text"
 
%18 = sitofp i32 %11 to double
%i328B

	full_text
	
i32 %11
Ffmul8B<
:
	full_text-
+
)%19 = fmul double %18, 0x3F9D41D41D41D41D
+double8B

	full_text


double %18
ogetelementptr8B\
Z
	full_textM
K
I%20 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
?call8B}
{
	full_textn
l
jcall void @exact_solution(double %19, double 0.000000e+00, double %17, double* nonnull %20, double* %1) #4
+double8B

	full_text


double %19
+double8B

	full_text


double %17
-double*8B

	full_text

double* %20
0shl8B'
%
	full_text

%21 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
1shl8B(
&
	full_text

%23 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
7mul8B.
,
	full_text

%25 = mul nsw i64 %22, 6845
%i648B

	full_text
	
i64 %22
4mul8B+
)
	full_text

%26 = mul nsw i64 %24, 5
%i648B

	full_text
	
i64 %24
2add8B)
'
	full_text

%27 = add i64 %25, %26
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %26
Ugetelementptr8BB
@
	full_text3
1
/%28 = getelementptr double, double* %0, i64 %27
%i648B

	full_text
	
i64 %27
@bitcast8B3
1
	full_text$
"
 %29 = bitcast double* %28 to i8*
-double*8B

	full_text

double* %28
tcall8Bj
h
	full_text[
Y
Wcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %29, i8* align 8 %7, i64 40, i1 false)
%i8*8B

	full_text
	
i8* %29
$i8*8B

	full_text


i8* %7
4add8B+
)
	full_text

%30 = add nsw i32 %3, -1
?call8B}
{
	full_textn
l
jcall void @exact_solution(double %19, double 1.000000e+00, double %17, double* nonnull %20, double* %1) #4
+double8B

	full_text


double %19
+double8B

	full_text


double %17
-double*8B

	full_text

double* %20
6sext8B,
*
	full_text

%31 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
6mul8B-
+
	full_text

%32 = mul nsw i64 %31, 185
%i648B

	full_text
	
i64 %31
2add8B)
'
	full_text

%33 = add i64 %25, %32
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %32
2add8B)
'
	full_text

%34 = add i64 %33, %26
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %26
Ugetelementptr8BB
@
	full_text3
1
/%35 = getelementptr double, double* %0, i64 %34
%i648B

	full_text
	
i64 %34
@bitcast8B3
1
	full_text$
"
 %36 = bitcast double* %35 to i8*
-double*8B

	full_text

double* %35
tcall8Bj
h
	full_text[
Y
Wcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %36, i8* align 8 %7, i64 40, i1 false)
%i8*8B

	full_text
	
i8* %36
$i8*8B

	full_text


i8* %7
'br8B

	full_text

br label %37
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %7) #4
$i8*8B

	full_text


i8* %7
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %2
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
:double8B,
*
	full_text

double 0x3F9D41D41D41D41D
4double8B&
$
	full_text

double 0.000000e+00
&i648B

	full_text


i64 6845
$i648B

	full_text


i64 40
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 5
4double8B&
$
	full_text

double 1.000000e+00
%i648B

	full_text
	
i64 185
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1        		 
 

                     ! "  #$ ## %& %% '( '' )* )) +, ++ -. -- /0 /1 // 23 22 45 44 67 68 66 99 :; :< := :: >? >> @A @@ BC BD BB EF EG EE HI HH JK JJ LM LN LL OQ PP RS T 2T HU 9V V :W    	  
      
      ! " $# &	 (' *% ,) .+ 0- 1/ 32 54 7 8 ; < =9 ?> A+ C@ DB F- GE IH KJ M N Q  PO P ZZ XX [[ YY \\ RP [[ P ZZ : ZZ : XX 6 \\ 6L \\ L	 YY 	 YY ] ] ^ _ +` ` 6` L` Pa #a %a 'a )b 	c 6c Ld -e :f @g 9h h i i "
initialize4"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
exact_solution"
llvm.lifetime.end.p0i8"
llvm.memcpy.p0i8.p0i8.i64*?
npb-SP-initialize4.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

devmap_label
 

wgsize
$

wgsize_log1p
 ??A

transfer_bytes
???	
 
transfer_bytes_log1p
 ??A