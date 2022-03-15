

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
3icmpB+
)
	full_text

%7 = icmp slt i32 %6, %2
"i32B

	full_text


i32 %6
6brB0
.
	full_text!

br i1 %7, label %8, label %74
 i1B

	full_text	

i1 %7
3mul8B*
(
	full_text

%9 = mul nsw i32 %6, %2
$i328B

	full_text


i32 %6
5icmp8B+
)
	full_text

%10 = icmp sgt i32 %3, 0
5sext8B+
)
	full_text

%11 = sext i32 %2 to i64
0shl8B'
%
	full_text

%12 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%13 = ashr exact i64 %12, 32
%i648B

	full_text
	
i64 %12
5sext8B+
)
	full_text

%14 = sext i32 %9 to i64
$i328B

	full_text


i32 %9
5zext8B+
)
	full_text

%15 = zext i32 %3 to i64
0and8B'
%
	full_text

%16 = and i64 %15, 1
%i648B

	full_text
	
i64 %15
4icmp8B*
(
	full_text

%17 = icmp eq i32 %3, 1
6sub8B-
+
	full_text

%18 = sub nsw i64 %15, %16
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %16
5icmp8B+
)
	full_text

%19 = icmp eq i64 %16, 0
%i648B

	full_text
	
i64 %16
'br8B

	full_text

br label %20
Cphi8B:
8
	full_text+
)
'%21 = phi i64 [ %13, %8 ], [ %72, %66 ]
%i648B

	full_text
	
i64 %13
%i648B

	full_text
	
i64 %72
6add8B-
+
	full_text

%22 = add nsw i64 %21, %14
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %14
\getelementptr8BI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %0, i64 %22
%i648B

	full_text
	
i64 %22
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %23, align 4, !tbaa !9
+float*8B

	full_text


float* %23
:br8B2
0
	full_text#
!
br i1 %10, label %24, label %66
#i18B

	full_text


i1 %10
:br8B2
0
	full_text#
!
br i1 %17, label %50, label %25
#i18B

	full_text


i1 %17
'br8B

	full_text

br label %26
Ophi8BF
D
	full_text7
5
3%27 = phi float [ 0.000000e+00, %25 ], [ %46, %26 ]
)float8B

	full_text

	float %46
Bphi8B9
7
	full_text*
(
&%28 = phi i64 [ 0, %25 ], [ %47, %26 ]
%i648B

	full_text
	
i64 %47
Dphi8B;
9
	full_text,
*
(%29 = phi i64 [ %18, %25 ], [ %48, %26 ]
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %48
6mul8B-
+
	full_text

%30 = mul nsw i64 %28, %11
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %11
6add8B-
+
	full_text

%31 = add nsw i64 %30, %13
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %13
\getelementptr8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %1, i64 %31
%i648B

	full_text
	
i64 %31
Lload8BB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !9
+float*8B

	full_text


float* %32
6add8B-
+
	full_text

%34 = add nsw i64 %30, %21
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %1, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !9
+float*8B

	full_text


float* %35
ecall8B[
Y
	full_textL
J
H%37 = tail call float @llvm.fmuladd.f32(float %33, float %36, float %27)
)float8B

	full_text

	float %33
)float8B

	full_text

	float %36
)float8B

	full_text

	float %27
Lstore8BA
?
	full_text2
0
.store float %37, float* %23, align 4, !tbaa !9
)float8B

	full_text

	float %37
+float*8B

	full_text


float* %23
.or8B&
$
	full_text

%38 = or i64 %28, 1
%i648B

	full_text
	
i64 %28
6mul8B-
+
	full_text

%39 = mul nsw i64 %38, %11
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %11
6add8B-
+
	full_text

%40 = add nsw i64 %39, %13
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %13
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !9
+float*8B

	full_text


float* %41
6add8B-
+
	full_text

%43 = add nsw i64 %39, %21
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %1, i64 %43
%i648B

	full_text
	
i64 %43
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !9
+float*8B

	full_text


float* %44
ecall8B[
Y
	full_textL
J
H%46 = tail call float @llvm.fmuladd.f32(float %42, float %45, float %37)
)float8B

	full_text

	float %42
)float8B

	full_text

	float %45
)float8B

	full_text

	float %37
Lstore8BA
?
	full_text2
0
.store float %46, float* %23, align 4, !tbaa !9
)float8B

	full_text

	float %46
+float*8B

	full_text


float* %23
4add8B+
)
	full_text

%47 = add nsw i64 %28, 2
%i648B

	full_text
	
i64 %28
1add8B(
&
	full_text

%48 = add i64 %29, -2
%i648B

	full_text
	
i64 %29
5icmp8B+
)
	full_text

%49 = icmp eq i64 %48, 0
%i648B

	full_text
	
i64 %48
:br8B2
0
	full_text#
!
br i1 %49, label %50, label %26
#i18B

	full_text


i1 %49
Hphi8B?
=
	full_text0
.
,%51 = phi float [ undef, %24 ], [ %46, %26 ]
)float8B

	full_text

	float %46
Ophi8BF
D
	full_text7
5
3%52 = phi float [ 0.000000e+00, %24 ], [ %46, %26 ]
)float8B

	full_text

	float %46
Bphi8B9
7
	full_text*
(
&%53 = phi i64 [ 0, %24 ], [ %47, %26 ]
%i648B

	full_text
	
i64 %47
:br8B2
0
	full_text#
!
br i1 %19, label %63, label %54
#i18B

	full_text


i1 %19
6mul8B-
+
	full_text

%55 = mul nsw i64 %53, %11
%i648B

	full_text
	
i64 %53
%i648B

	full_text
	
i64 %11
6add8B-
+
	full_text

%56 = add nsw i64 %55, %13
%i648B

	full_text
	
i64 %55
%i648B

	full_text
	
i64 %13
\getelementptr8BI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %1, i64 %56
%i648B

	full_text
	
i64 %56
Lload8BB
@
	full_text3
1
/%58 = load float, float* %57, align 4, !tbaa !9
+float*8B

	full_text


float* %57
6add8B-
+
	full_text

%59 = add nsw i64 %55, %21
%i648B

	full_text
	
i64 %55
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %1, i64 %59
%i648B

	full_text
	
i64 %59
Lload8BB
@
	full_text3
1
/%61 = load float, float* %60, align 4, !tbaa !9
+float*8B

	full_text


float* %60
ecall8B[
Y
	full_textL
J
H%62 = tail call float @llvm.fmuladd.f32(float %58, float %61, float %52)
)float8B

	full_text

	float %58
)float8B

	full_text

	float %61
)float8B

	full_text

	float %52
Lstore8BA
?
	full_text2
0
.store float %62, float* %23, align 4, !tbaa !9
)float8B

	full_text

	float %62
+float*8B

	full_text


float* %23
'br8B

	full_text

br label %63
Fphi8B=
;
	full_text.
,
*%64 = phi float [ %51, %50 ], [ %62, %54 ]
)float8B

	full_text

	float %51
)float8B

	full_text

	float %62
>bitcast8B1
/
	full_text"
 
%65 = bitcast float %64 to i32
)float8B

	full_text

	float %64
'br8B

	full_text

br label %66
Bphi8	B9
7
	full_text*
(
&%67 = phi i32 [ %65, %63 ], [ 0, %20 ]
%i328	B

	full_text
	
i32 %65
6mul8	B-
+
	full_text

%68 = mul nsw i64 %21, %11
%i648	B

	full_text
	
i64 %21
%i648	B

	full_text
	
i64 %11
6add8	B-
+
	full_text

%69 = add nsw i64 %68, %13
%i648	B

	full_text
	
i64 %68
%i648	B

	full_text
	
i64 %13
\getelementptr8	BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %0, i64 %69
%i648	B

	full_text
	
i64 %69
@bitcast8	B3
1
	full_text$
"
 %71 = bitcast float* %70 to i32*
+float*8	B

	full_text


float* %70
Hstore8	B=
;
	full_text.
,
*store i32 %67, i32* %71, align 4, !tbaa !9
%i328	B

	full_text
	
i32 %67
'i32*8	B

	full_text


i32* %71
4add8	B+
)
	full_text

%72 = add nsw i64 %21, 1
%i648	B

	full_text
	
i64 %21
7icmp8	B-
+
	full_text

%73 = icmp eq i64 %72, %11
%i648	B

	full_text
	
i64 %72
%i648	B

	full_text
	
i64 %11
:br8	B2
0
	full_text#
!
br i1 %73, label %74, label %20
#i18	B

	full_text


i1 %73
$ret8
B

	full_text


ret void
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
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
$i648B

	full_text


i64 32
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 -2
+float8B

	full_text

float undef
#i648B

	full_text	

i64 0      	  

                      !  "# "" $% $$ &' &) (, ++ -. -- /0 /1 // 23 24 22 56 57 55 89 88 :; :: <= <> << ?@ ?? AB AA CD CE CF CC GH GI GG JK JJ LM LN LL OP OQ OO RS RR TU TT VW VX VV YZ YY [\ [[ ]^ ]_ ]` ]] ab ac aa de dd fg ff hi hh jk jm ll no nn pq pp rs ru tv tt wx wy ww z{ zz |} || ~ ~	€ ~~ 
‚  ƒ„ ƒƒ …† …
‡ …
ˆ …… ‰Š ‰
‹ ‰‰ ŒŽ 
  ‘  ’” ““ •– •
— •• ˜™ ˜
š ˜˜ ›
œ ›› ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §ª "ª ›« 8« ?« R« Y« z« 	¬ 	¬ ¬ ­ 
­ ­     	        ¢    ! #" %
 ' )] ,d . 0f 1- 3 42 6 75 98 ;2 = >< @? B: DA E+ FC H" I- KJ M NL P QO SR UL W XV ZY \T ^[ _C `] b" c- e/ gf ih k] m] od q sp u vt x yw {z }t  €~ ‚ „| †ƒ ‡n ˆ… Š" ‹l Ž…  ‘ ” – —• ™ š˜ œ› ž“   ¡ £¢ ¥ ¦¤ ¨  © & (& “( l( *§ ©§ r r t* +’ “Œ j lj + © ®® ¯¯C ¯¯ C] ¯¯ ]… ¯¯ … ®® 	° 	° ± $± +± n² 	² 

² “	³ d	´ 	´ J
´ ¢	µ 	¶ f· l	¸ ¸ -	¸ h¸ p"
covar_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*¥
,polybench-gpu-1.0-covariance-covar_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
££ŠA

wgsize
€

devmap_label


transfer_bytes
ŒÀ‚

wgsize_log1p
££ŠA