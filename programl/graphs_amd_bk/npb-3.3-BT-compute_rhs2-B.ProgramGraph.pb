

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 2) #2
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
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #2
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
4icmpB,
*
	full_text

%11 = icmp slt i32 %7, %4
"i32B

	full_text


i32 %7
5truncB,
*
	full_text

%12 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %3
#i32B

	full_text
	
i32 %12
/andB(
&
	full_text

%14 = and i1 %11, %13
!i1B

	full_text


i1 %11
!i1B

	full_text


i1 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %10, %2
#i32B

	full_text
	
i32 %10
/andB(
&
	full_text

%16 = and i1 %14, %15
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %15
8brB2
0
	full_text#
!
br i1 %16, label %17, label %51
!i1B

	full_text


i1 %16
Ybitcast8BL
J
	full_text=
;
9%18 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
Ybitcast8BL
J
	full_text=
;
9%19 = bitcast double* %1 to [103 x [103 x [5 x double]]]*
0shl8B'
%
	full_text

%20 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%21 = ashr exact i64 %20, 32
%i648B

	full_text
	
i64 %20
0shl8B'
%
	full_text

%22 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
0shl8B'
%
	full_text

%24 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
®getelementptr8Bî
ë
	full_textÉ
Ä
~%26 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%27 = bitcast double* %26 to i64*
-double*8B

	full_text

double* %26
Hload8B>
<
	full_text/
-
+%28 = load i64, i64* %27, align 8, !tbaa !8
'i64*8B

	full_text


i64* %27
®getelementptr8Bî
ë
	full_textÉ
Ä
~%29 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%30 = bitcast double* %29 to i64*
-double*8B

	full_text

double* %29
Hstore8B=
;
	full_text.
,
*store i64 %28, i64* %30, align 8, !tbaa !8
%i648B

	full_text
	
i64 %28
'i64*8B

	full_text


i64* %30
®getelementptr8Bî
ë
	full_textÉ
Ä
~%31 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%32 = bitcast double* %31 to i64*
-double*8B

	full_text

double* %31
Hload8B>
<
	full_text/
-
+%33 = load i64, i64* %32, align 8, !tbaa !8
'i64*8B

	full_text


i64* %32
®getelementptr8Bî
ë
	full_textÉ
Ä
~%34 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%35 = bitcast double* %34 to i64*
-double*8B

	full_text

double* %34
Hstore8B=
;
	full_text.
,
*store i64 %33, i64* %35, align 8, !tbaa !8
%i648B

	full_text
	
i64 %33
'i64*8B

	full_text


i64* %35
®getelementptr8Bî
ë
	full_textÉ
Ä
~%36 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%37 = bitcast double* %36 to i64*
-double*8B

	full_text

double* %36
Hload8B>
<
	full_text/
-
+%38 = load i64, i64* %37, align 8, !tbaa !8
'i64*8B

	full_text


i64* %37
®getelementptr8Bî
ë
	full_textÉ
Ä
~%39 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%40 = bitcast double* %39 to i64*
-double*8B

	full_text

double* %39
Hstore8B=
;
	full_text.
,
*store i64 %38, i64* %40, align 8, !tbaa !8
%i648B

	full_text
	
i64 %38
'i64*8B

	full_text


i64* %40
®getelementptr8Bî
ë
	full_textÉ
Ä
~%41 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%42 = bitcast double* %41 to i64*
-double*8B

	full_text

double* %41
Hload8B>
<
	full_text/
-
+%43 = load i64, i64* %42, align 8, !tbaa !8
'i64*8B

	full_text


i64* %42
®getelementptr8Bî
ë
	full_textÉ
Ä
~%44 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%45 = bitcast double* %44 to i64*
-double*8B

	full_text

double* %44
Hstore8B=
;
	full_text.
,
*store i64 %43, i64* %45, align 8, !tbaa !8
%i648B

	full_text
	
i64 %43
'i64*8B

	full_text


i64* %45
®getelementptr8Bî
ë
	full_textÉ
Ä
~%46 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%47 = bitcast double* %46 to i64*
-double*8B

	full_text

double* %46
Hload8B>
<
	full_text/
-
+%48 = load i64, i64* %47, align 8, !tbaa !8
'i64*8B

	full_text


i64* %47
®getelementptr8Bî
ë
	full_textÉ
Ä
~%49 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%50 = bitcast double* %49 to i64*
-double*8B

	full_text

double* %49
Hstore8B=
;
	full_text.
,
*store i64 %48, i64* %50, align 8, !tbaa !8
%i648B

	full_text
	
i64 %48
'i64*8B

	full_text


i64* %50
'br8B

	full_text

br label %51
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %2
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
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 2       	  
 

                     !    "# "" $% $$ &' &( &) &* && +, ++ -. -- /0 /1 /2 /3 // 45 44 67 68 66 9: 9; 9< 9= 99 >? >> @A @@ BC BD BE BF BB GH GG IJ IK II LM LN LO LP LL QR QQ ST SS UV UW UX UY UU Z[ ZZ \] \^ \\ _` _a _b _c __ de dd fg ff hi hj hk hl hh mn mm op oq oo rs rt ru rv rr wx ww yz yy {| {} {~ { {{ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ Ö	á 	à 	â ä ã    	 
           ! #" % ' (  )$ *& ,+ . 0 1  2$ 3/ 5- 74 8 : ;  <$ =9 ?> A C D  E$ FB H@ JG K M N  O$ PL RQ T V W  X$ YU [S ]Z ^ ` a  b$ c_ ed g i j  k$ lh nf pm q s t  u$ vr xw z | }  ~$ { Åy ÉÄ Ñ  ÜÖ Ü åå Ü åå  åå  åå 	ç &	ç /	é 	é 	é 	é  	é "	é $è ê 	ë r	ë {	í L	í U	ì _	ì h	î 9	î Bï "
compute_rhs2"
_Z13get_global_idj*ë
npb-BT-compute_rhs2_B.clu
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

wgsize_log1p
 ¯ßA
 
transfer_bytes_log1p
 ¯ßA

transfer_bytes	
òÆ¥Ú

wgsize
 