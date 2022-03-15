

[external]
JcallBB
@
	full_text3
1
/%6 = tail call i64 @_Z12get_local_idj(i32 0) #3
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
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
KcallBC
A
	full_text4
2
0%10 = tail call i64 @_Z12get_group_idj(i32 0) #3
1shlB*
(
	full_text

%11 = shl nsw i32 %9, 1
"i32B

	full_text


i32 %9
4sextB,
*
	full_text

%12 = sext i32 %11 to i64
#i32B

	full_text
	
i32 %11
ZgetelementptrBI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %1, i64 %12
#i64B

	full_text
	
i64 %12
>bitcastB3
1
	full_text$
"
 %14 = bitcast float* %13 to i32*
)float*B

	full_text


float* %13
FloadB>
<
	full_text/
-
+%15 = load i32, i32* %14, align 4, !tbaa !8
%i32*B

	full_text


i32* %14
1shlB*
(
	full_text

%16 = shl nsw i32 %7, 1
"i32B

	full_text


i32 %7
4sextB,
*
	full_text

%17 = sext i32 %16 to i64
#i32B

	full_text
	
i32 %16
ZgetelementptrBI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %2, i64 %17
#i64B

	full_text
	
i64 %17
>bitcastB3
1
	full_text$
"
 %19 = bitcast float* %18 to i32*
)float*B

	full_text


float* %18
FstoreB=
;
	full_text.
,
*store i32 %15, i32* %19, align 4, !tbaa !8
#i32B

	full_text
	
i32 %15
%i32*B

	full_text


i32* %19
,orB&
$
	full_text

%20 = or i32 %11, 1
#i32B

	full_text
	
i32 %11
4sextB,
*
	full_text

%21 = sext i32 %20 to i64
#i32B

	full_text
	
i32 %20
ZgetelementptrBI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %1, i64 %21
#i64B

	full_text
	
i64 %21
>bitcastB3
1
	full_text$
"
 %23 = bitcast float* %22 to i32*
)float*B

	full_text


float* %22
FloadB>
<
	full_text/
-
+%24 = load i32, i32* %23, align 4, !tbaa !8
%i32*B

	full_text


i32* %23
,orB&
$
	full_text

%25 = or i32 %16, 1
#i32B

	full_text
	
i32 %16
4sextB,
*
	full_text

%26 = sext i32 %25 to i64
#i32B

	full_text
	
i32 %25
ZgetelementptrBI
G
	full_text:
8
6%27 = getelementptr inbounds float, float* %2, i64 %26
#i64B

	full_text
	
i64 %26
>bitcastB3
1
	full_text$
"
 %28 = bitcast float* %27 to i32*
)float*B

	full_text


float* %27
FstoreB=
;
	full_text.
,
*store i32 %24, i32* %28, align 4, !tbaa !8
#i32B

	full_text
	
i32 %24
%i32*B

	full_text


i32* %28
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
IloadBA
?
	full_text2
0
.%29 = load float, float* %2, align 4, !tbaa !8
3icmpB+
)
	full_text

%30 = icmp ugt i32 %3, 1
8brB2
0
	full_text#
!
br i1 %30, label %31, label %35
!i1B

	full_text


i1 %30
Zgetelementptr8BG
E
	full_text8
6
4%32 = getelementptr inbounds float, float* %2, i64 1
Lload8BB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !8
+float*8B

	full_text


float* %32
6fadd8B,
*
	full_text

%34 = fadd float %29, %33
)float8B

	full_text

	float %29
)float8B

	full_text

	float %33
'br8B

	full_text

br label %46
0add8B'
%
	full_text

%36 = add i32 %3, -1
6zext8B,
*
	full_text

%37 = zext i32 %36 to i64
%i328B

	full_text
	
i32 %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %2, i64 %37
%i648B

	full_text
	
i64 %37
@bitcast8B3
1
	full_text$
"
 %39 = bitcast float* %38 to i32*
+float*8B

	full_text


float* %38
Hload8B>
<
	full_text/
-
+%40 = load i32, i32* %39, align 4, !tbaa !8
'i32*8B

	full_text


i32* %39
1shl8B(
&
	full_text

%41 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%42 = ashr exact i64 %41, 32
%i648B

	full_text
	
i64 %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %4, i64 %42
%i648B

	full_text
	
i64 %42
@bitcast8B3
1
	full_text$
"
 %44 = bitcast float* %43 to i32*
+float*8B

	full_text


float* %43
Hstore8B=
;
	full_text.
,
*store i32 %40, i32* %44, align 4, !tbaa !8
%i328B

	full_text
	
i32 %40
'i32*8B

	full_text


i32* %44
4icmp8B*
(
	full_text

%45 = icmp eq i32 %7, 0
$i328B

	full_text


i32 %7
:br8B2
0
	full_text#
!
br i1 %45, label %69, label %71
#i18B

	full_text


i1 %45
Bphi8B9
7
	full_text*
(
&%47 = phi i32 [ 1, %31 ], [ %67, %64 ]
%i328B

	full_text
	
i32 %67
Fphi8B=
;
	full_text.
,
*%48 = phi float [ %34, %31 ], [ %66, %64 ]
)float8B

	full_text

	float %34
)float8B

	full_text

	float %66
Fphi8B=
;
	full_text.
,
*%49 = phi float [ %29, %31 ], [ %65, %64 ]
)float8B

	full_text

	float %29
)float8B

	full_text

	float %65
8icmp8B.
,
	full_text

%50 = icmp slt i32 %16, %47
%i328B

	full_text
	
i32 %16
%i328B

	full_text
	
i32 %47
:br8B2
0
	full_text#
!
br i1 %50, label %64, label %51
#i18B

	full_text


i1 %50
6sub8B-
+
	full_text

%52 = sub nsw i32 %16, %47
%i328B

	full_text
	
i32 %16
%i328B

	full_text
	
i32 %47
6sext8B,
*
	full_text

%53 = sext i32 %52 to i64
%i328B

	full_text
	
i32 %52
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %2, i64 %53
%i648B

	full_text
	
i64 %53
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !8
+float*8B

	full_text


float* %54
Lload8BB
@
	full_text3
1
/%56 = load float, float* %18, align 4, !tbaa !8
+float*8B

	full_text


float* %18
6fadd8B,
*
	full_text

%57 = fadd float %55, %56
)float8B

	full_text

	float %55
)float8B

	full_text

	float %56
6sub8B-
+
	full_text

%58 = sub nsw i32 %25, %47
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %47
6sext8B,
*
	full_text

%59 = sext i32 %58 to i64
%i328B

	full_text
	
i32 %58
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %2, i64 %59
%i648B

	full_text
	
i64 %59
Lload8BB
@
	full_text3
1
/%61 = load float, float* %60, align 4, !tbaa !8
+float*8B

	full_text


float* %60
Lload8BB
@
	full_text3
1
/%62 = load float, float* %27, align 4, !tbaa !8
+float*8B

	full_text


float* %27
6fadd8B,
*
	full_text

%63 = fadd float %61, %62
)float8B

	full_text

	float %61
)float8B

	full_text

	float %62
'br8B

	full_text

br label %64
Fphi8B=
;
	full_text.
,
*%65 = phi float [ %57, %51 ], [ %49, %46 ]
)float8B

	full_text

	float %57
)float8B

	full_text

	float %49
Fphi8B=
;
	full_text.
,
*%66 = phi float [ %63, %51 ], [ %48, %46 ]
)float8B

	full_text

	float %63
)float8B

	full_text

	float %48
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
Lstore8BA
?
	full_text2
0
.store float %65, float* %18, align 4, !tbaa !8
)float8B

	full_text

	float %65
+float*8B

	full_text


float* %18
Lstore8BA
?
	full_text2
0
.store float %66, float* %27, align 4, !tbaa !8
)float8B

	full_text

	float %66
+float*8B

	full_text


float* %27
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
4shl8B+
)
	full_text

%67 = shl nsw i32 %47, 1
%i328B

	full_text
	
i32 %47
7icmp8B-
+
	full_text

%68 = icmp ult i32 %67, %3
%i328B

	full_text
	
i32 %67
:br8B2
0
	full_text#
!
br i1 %68, label %46, label %35
#i18B

	full_text


i1 %68
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %0, i64 %12
%i648B

	full_text
	
i64 %12
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %70, align 4, !tbaa !8
+float*8B

	full_text


float* %70
'br8B

	full_text

br label %79
5add8B,
*
	full_text

%72 = add nsw i32 %16, -1
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%73 = sext i32 %72 to i64
%i328B

	full_text
	
i32 %72
\getelementptr8BI
G
	full_text:
8
6%74 = getelementptr inbounds float, float* %2, i64 %73
%i648B

	full_text
	
i64 %73
@bitcast8B3
1
	full_text$
"
 %75 = bitcast float* %74 to i32*
+float*8B

	full_text


float* %74
Hload8B>
<
	full_text/
-
+%76 = load i32, i32* %75, align 4, !tbaa !8
'i32*8B

	full_text


i32* %75
\getelementptr8BI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %0, i64 %12
%i648B

	full_text
	
i64 %12
@bitcast8B3
1
	full_text$
"
 %78 = bitcast float* %77 to i32*
+float*8B

	full_text


float* %77
Hstore8B=
;
	full_text.
,
*store i32 %76, i32* %78, align 4, !tbaa !8
%i328B

	full_text
	
i32 %76
'i32*8B

	full_text


i32* %78
'br8B

	full_text

br label %79
Hload8B>
<
	full_text/
-
+%80 = load i32, i32* %19, align 4, !tbaa !8
'i32*8B

	full_text


i32* %19
\getelementptr8BI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %0, i64 %21
%i648B

	full_text
	
i64 %21
@bitcast8B3
1
	full_text$
"
 %82 = bitcast float* %81 to i32*
+float*8B

	full_text


float* %81
Hstore8B=
;
	full_text.
,
*store i32 %80, i32* %82, align 4, !tbaa !8
%i328B

	full_text
	
i32 %80
'i32*8B

	full_text


i32* %82
$ret8B

	full_text


ret void
$i328	B

	full_text


i32 %3
*float*8	B

	full_text

	float* %2
*float*8	B

	full_text

	float* %4
*float*8	B

	full_text

	float* %0
*float*8	B

	full_text

	float* %1
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
$i328	B

	full_text


i32 -1
$i648	B

	full_text


i64 32
#i648	B

	full_text	

i64 1
2float8	B%
#
	full_text

float 0.000000e+00
#i328	B

	full_text	

i32 1
#i328	B

	full_text	

i32 0       	  
 

                       !" !! #$ ## %& %% '( '' )* )) +, ++ -. -- /0 /1 // 22 33 44 56 57 89 88 :; :< :: => ?@ ?? AB AA CD CC EF EE GH GG IJ II KL KK MN MM OP OQ OO RS RR TU TW VV XY XZ XX [\ [] [[ ^_ ^` ^^ ab ad ce cc fg ff hi hh jk jj lm ll no np nn qr qs qq tu tt vw vv xy xx z{ zz |} |~ ||  €
‚ €€ ƒ„ ƒ
… ƒƒ †† ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ    ‘  ’“ ’
• ”” –
— –– ˜š ™™ ›œ ›› 
  Ÿ  ŸŸ ¡¢ ¡¡ £
¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª¬ «« ­
® ­­ ¯° ¯¯ ±² ±
³ ±± ´µ 4µ >
µ ¶ ¶ +¶ 3¶ 7¶ A¶ h¶ v¶ · K¸ ”¸ £¸ ­¹ ¹ !   	 
             "! $# & (' *) ,+ .% 0- 14 67 93 ;8 <> @? BA DC F HG JI LK NE PM Q SR U W: Yƒ Z3 \€ ] _V `^ b dV ec gf ih k mj ol p' rV sq ut wv y+ {x }z ~n [ ‚| „X …€ ˆ ‰ƒ ‹+ ŒV  ‘ “
 •” — š™ œ›   Ÿ ¢
 ¤£ ¦¡ ¨¥ © ¬ ®­ °« ²¯ ³5 75 >= VT ”T ™a €a c˜ «ª «’ V’ > € ºº »» ´ ¼¼ ½½† ½½ †2 ½½ 2 »»  ½½  ºº  ¼¼ 	¾ >
¾ ™	¿ G	¿ I	À 7Á –	Â 	Â 	Â 	Â 'Â 2	Â 4Â VÂ †Â 
Â Ã Ã Ã 	Ã R"
ScanLargeArrays"
_Z12get_local_idj"
_Z13get_global_idj"
_Z12get_group_idj"
_Z7barrierj*›
"ScanLargeArrays-ScanLargeArrays.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282€

wgsize_log1p
W°GA
 
transfer_bytes_log1p
W°GA

transfer_bytes
€ˆ

devmap_label
 

wgsize
€