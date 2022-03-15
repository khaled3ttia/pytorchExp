

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 1) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
3icmpB+
)
	full_text

%8 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
4truncB+
)
	full_text

%9 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %2
"i32B

	full_text


i32 %9
.andB'
%
	full_text

%11 = and i1 %8, %10
 i1B

	full_text	

i1 %8
!i1B

	full_text


i1 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %48
!i1B

	full_text


i1 %11
Ybitcast8BL
J
	full_text=
;
9%13 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
5icmp8B+
)
	full_text

%14 = icmp sgt i32 %1, 0
:br8B2
0
	full_text#
!
br i1 %14, label %15, label %48
#i18B

	full_text


i1 %14
0shl8B'
%
	full_text

%16 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
0shl8B'
%
	full_text

%18 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%19 = ashr exact i64 %18, 32
%i648B

	full_text
	
i64 %18
5zext8B+
)
	full_text

%20 = zext i32 %1 to i64
0and8B'
%
	full_text

%21 = and i64 %20, 1
%i648B

	full_text
	
i64 %20
4icmp8B*
(
	full_text

%22 = icmp eq i32 %1, 1
:br8B2
0
	full_text#
!
br i1 %22, label %40, label %23
#i18B

	full_text


i1 %22
6sub8B-
+
	full_text

%24 = sub nsw i64 %20, %21
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %21
'br8B

	full_text

br label %25
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %23 ], [ %37, %25 ]
%i648B

	full_text
	
i64 %37
Dphi8B;
9
	full_text,
*
(%27 = phi i64 [ %24, %23 ], [ %38, %25 ]
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %38
®getelementptr8Bî
ë
	full_textÉ
Ä
~%28 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %26, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %26
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %28, align 8, !tbaa !8
-double*8B

	full_text

double* %28
®getelementptr8Bî
ë
	full_textÉ
Ä
~%29 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %26, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %26
®getelementptr8Bî
ë
	full_textÉ
Ä
~%30 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %26, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %26
@bitcast8B3
1
	full_text$
"
 %31 = bitcast double* %29 to i8*
-double*8B

	full_text

double* %29
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %31, i8 0, i64 24, i1 false)
%i8*8B

	full_text
	
i8* %31
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
.or8B&
$
	full_text

%32 = or i64 %26, 1
%i648B

	full_text
	
i64 %26
®getelementptr8Bî
ë
	full_textÉ
Ä
~%33 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %32, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %32
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
®getelementptr8Bî
ë
	full_textÉ
Ä
~%34 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %32, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %32
®getelementptr8Bî
ë
	full_textÉ
Ä
~%35 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %32, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %32
@bitcast8B3
1
	full_text$
"
 %36 = bitcast double* %34 to i8*
-double*8B

	full_text

double* %34
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %36, i8 0, i64 24, i1 false)
%i8*8B

	full_text
	
i8* %36
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
4add8B+
)
	full_text

%37 = add nsw i64 %26, 2
%i648B

	full_text
	
i64 %26
1add8B(
&
	full_text

%38 = add i64 %27, -2
%i648B

	full_text
	
i64 %27
5icmp8B+
)
	full_text

%39 = icmp eq i64 %38, 0
%i648B

	full_text
	
i64 %38
:br8B2
0
	full_text#
!
br i1 %39, label %40, label %25
#i18B

	full_text


i1 %39
Bphi8B9
7
	full_text*
(
&%41 = phi i64 [ 0, %15 ], [ %37, %25 ]
%i648B

	full_text
	
i64 %37
5icmp8B+
)
	full_text

%42 = icmp eq i64 %21, 0
%i648B

	full_text
	
i64 %21
:br8B2
0
	full_text#
!
br i1 %42, label %48, label %43
#i18B

	full_text


i1 %42
®getelementptr8Bî
ë
	full_textÉ
Ä
~%44 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %41, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %41
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
®getelementptr8Bî
ë
	full_textÉ
Ä
~%45 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %41, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %41
®getelementptr8Bî
ë
	full_textÉ
Ä
~%46 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %41, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %41
@bitcast8B3
1
	full_text$
"
 %47 = bitcast double* %45 to i8*
-double*8B

	full_text

double* %45
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %47, i8 0, i64 24, i1 false)
%i8*8B

	full_text
	
i8* %47
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
'br8B

	full_text

br label %48
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %1
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
#i648B

	full_text	

i64 4
!i88B

	full_text

i8 0
%i18B

	full_text


i1 false
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 1.000000e+00
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 -2
$i648B

	full_text


i64 24
#i648B

	full_text	

i64 0        	
 		                     !  # "$ "" %' && () (* (( +, +- +. +/ ++ 01 00 23 24 25 26 22 78 79 7: 7; 77 <= << >? >> @A @@ BC BB DE DF DG DH DD IJ II KL KM KN KO KK PQ PR PS PT PP UV UU WX WW YZ YY [\ [[ ]^ ]] _` __ ab ad cc ef ee gh gj ik il im ii no nn pq pr ps pt pp uv uw ux uy uu z{ zz |} || ~ ~~ Ä	Ç 	É 	Ñ Ö Ö Ö     
 	         ! # $[ '" )] * , - .& /+ 1 3 4 5& 6 8 9 :& ;2 =< ?7 A& C E F GB HD J L M NB O Q R SB TK VU XP Z& \( ^] `_ b[ d fe h j k lc mi o q r sc t v w xc yp {z }u   Å  Å  c  "g Åg i% &Ä Åa ca & Å ÜÜ áá ÜÜ  ÜÜ W áá W> áá >| áá |	à 7	à P	à u	â >	â W	â |	ä >	ä W	ä |	ã 	ã 	ã 	ã 	å 	å 2	å B	å K	å pç 	ç 	é [è 0è @è Iè Yè nè ~ê 	ê 	ë ]	í >	í W	í |ì &	ì +	ì D	ì _ì c	ì e	ì i"
initialize1"
_Z13get_global_idj"
llvm.memset.p0i8.i64*é
npb-SP-initialize1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

devmap_label

 
transfer_bytes_log1p
é™A

wgsize_log1p
é™A

transfer_bytes	
∞æ·

wgsize
@