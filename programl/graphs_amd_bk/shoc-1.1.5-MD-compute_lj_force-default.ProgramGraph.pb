

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
6andB/
-
	full_text 

%11 = and i64 %9, 4294967295
"i64B

	full_text


i64 %9
fgetelementptrBU
S
	full_textF
D
B%12 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %11
#i64B

	full_text
	
i64 %11
WloadBO
M
	full_text@
>
<%13 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %12
3icmpB+
)
	full_text

%14 = icmp sgt i32 %2, 0
8brB2
0
	full_text#
!
br i1 %14, label %15, label %63
!i1B

	full_text


i1 %14
Rextractelement8B>
<
	full_text/
-
+%16 = extractelement <4 x float> %13, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %13
Rextractelement8B>
<
	full_text/
-
+%17 = extractelement <4 x float> %13, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %13
Rextractelement8B>
<
	full_text/
-
+%18 = extractelement <4 x float> %13, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %13
?fsub8B5
3
	full_text&
$
"%19 = fsub float -0.000000e+00, %6
5zext8B+
)
	full_text

%20 = zext i32 %2 to i64
'br8B

	full_text

br label %21
Bphi8B9
7
	full_text*
(
&%22 = phi i64 [ 0, %15 ], [ %61, %59 ]
%i648B

	full_text
	
i64 %61
Xphi8BO
M
	full_text@
>
<%23 = phi <4 x float> [ zeroinitializer, %15 ], [ %60, %59 ]
5<4 x float>8B"
 
	full_text

<4 x float> %60
8trunc8B-
+
	full_text

%24 = trunc i64 %22 to i32
%i648B

	full_text
	
i64 %22
5mul8B,
*
	full_text

%25 = mul nsw i32 %24, %7
%i328B

	full_text
	
i32 %24
2add8B)
'
	full_text

%26 = add i32 %25, %10
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %10
6zext8B,
*
	full_text

%27 = zext i32 %26 to i64
%i328B

	full_text
	
i32 %26
Xgetelementptr8BE
C
	full_text6
4
2%28 = getelementptr inbounds i32, i32* %3, i64 %27
%i648B

	full_text
	
i64 %27
Iload8B?
=
	full_text0
.
,%29 = load i32, i32* %28, align 4, !tbaa !12
'i32*8B

	full_text


i32* %28
6sext8B,
*
	full_text

%30 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
hgetelementptr8BU
S
	full_textF
D
B%31 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %30
%i648B

	full_text
	
i64 %30
Yload8BO
M
	full_text@
>
<%32 = load <4 x float>, <4 x float>* %31, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %31
Rextractelement8B>
<
	full_text/
-
+%33 = extractelement <4 x float> %32, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %32
6fsub8B,
*
	full_text

%34 = fsub float %16, %33
)float8B

	full_text

	float %16
)float8B

	full_text

	float %33
Rextractelement8B>
<
	full_text/
-
+%35 = extractelement <4 x float> %32, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %32
6fsub8B,
*
	full_text

%36 = fsub float %17, %35
)float8B

	full_text

	float %17
)float8B

	full_text

	float %35
Rextractelement8B>
<
	full_text/
-
+%37 = extractelement <4 x float> %32, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %32
6fsub8B,
*
	full_text

%38 = fsub float %18, %37
)float8B

	full_text

	float %18
)float8B

	full_text

	float %37
6fmul8B,
*
	full_text

%39 = fmul float %36, %36
)float8B

	full_text

	float %36
)float8B

	full_text

	float %36
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %34, float %34, float %39)
)float8B

	full_text

	float %34
)float8B

	full_text

	float %34
)float8B

	full_text

	float %39
ecall8B[
Y
	full_textL
J
H%41 = tail call float @llvm.fmuladd.f32(float %38, float %38, float %40)
)float8B

	full_text

	float %38
)float8B

	full_text

	float %38
)float8B

	full_text

	float %40
9fcmp8B/
-
	full_text 

%42 = fcmp olt float %41, %4
)float8B

	full_text

	float %41
:br8B2
0
	full_text#
!
br i1 %42, label %43, label %59
#i18B

	full_text


i1 %42
Lfdiv8BB
@
	full_text3
1
/%44 = fdiv float 1.000000e+00, %41, !fpmath !14
)float8B

	full_text

	float %41
6fmul8B,
*
	full_text

%45 = fmul float %44, %44
)float8B

	full_text

	float %44
)float8B

	full_text

	float %44
6fmul8B,
*
	full_text

%46 = fmul float %44, %45
)float8B

	full_text

	float %44
)float8B

	full_text

	float %45
6fmul8B,
*
	full_text

%47 = fmul float %44, %46
)float8B

	full_text

	float %44
)float8B

	full_text

	float %46
dcall8BZ
X
	full_textK
I
G%48 = tail call float @llvm.fmuladd.f32(float %5, float %46, float %19)
)float8B

	full_text

	float %46
)float8B

	full_text

	float %19
6fmul8B,
*
	full_text

%49 = fmul float %47, %48
)float8B

	full_text

	float %47
)float8B

	full_text

	float %48
Rextractelement8B>
<
	full_text/
-
+%50 = extractelement <4 x float> %23, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %23
ecall8B[
Y
	full_textL
J
H%51 = tail call float @llvm.fmuladd.f32(float %34, float %49, float %50)
)float8B

	full_text

	float %34
)float8B

	full_text

	float %49
)float8B

	full_text

	float %50
[insertelement8BH
F
	full_text9
7
5%52 = insertelement <4 x float> %23, float %51, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %23
)float8B

	full_text

	float %51
Rextractelement8B>
<
	full_text/
-
+%53 = extractelement <4 x float> %23, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %23
ecall8B[
Y
	full_textL
J
H%54 = tail call float @llvm.fmuladd.f32(float %36, float %49, float %53)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %49
)float8B

	full_text

	float %53
[insertelement8BH
F
	full_text9
7
5%55 = insertelement <4 x float> %52, float %54, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %52
)float8B

	full_text

	float %54
Rextractelement8B>
<
	full_text/
-
+%56 = extractelement <4 x float> %23, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %23
ecall8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %38, float %49, float %56)
)float8B

	full_text

	float %38
)float8B

	full_text

	float %49
)float8B

	full_text

	float %56
[insertelement8BH
F
	full_text9
7
5%58 = insertelement <4 x float> %55, float %57, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %55
)float8B

	full_text

	float %57
'br8B

	full_text

br label %59
Lphi8BC
A
	full_text4
2
0%60 = phi <4 x float> [ %58, %43 ], [ %23, %21 ]
5<4 x float>8B"
 
	full_text

<4 x float> %58
5<4 x float>8B"
 
	full_text

<4 x float> %23
8add8B/
-
	full_text 

%61 = add nuw nsw i64 %22, 1
%i648B

	full_text
	
i64 %22
7icmp8B-
+
	full_text

%62 = icmp eq i64 %61, %20
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %20
:br8B2
0
	full_text#
!
br i1 %62, label %63, label %21
#i18B

	full_text


i1 %62
Wphi8BN
L
	full_text?
=
;%64 = phi <4 x float> [ zeroinitializer, %8 ], [ %60, %59 ]
5<4 x float>8B"
 
	full_text

<4 x float> %60
hgetelementptr8BU
S
	full_textF
D
B%65 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %11
%i648B

	full_text
	
i64 %11
Ystore8BN
L
	full_text?
=
;store <4 x float> %64, <4 x float>* %65, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %64
7<4 x float>*8B#
!
	full_text

<4 x float>* %65
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %2
(float8B

	full_text


float %6
6<4 x float>*8B"
 
	full_text

<4 x float>* %1
(float8B

	full_text


float %4
$i328B

	full_text


i32 %7
6<4 x float>*8B"
 
	full_text

<4 x float>* %0
(float8B

	full_text


float %5
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
A<4 x float>8B.
,
	full_text

<4 x float> zeroinitializer
,i648B!

	full_text

i64 4294967295
3float8B&
$
	full_text

float -0.000000e+00
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2
2float8B%
#
	full_text

float 1.000000e+00
#i648B

	full_text	

i64 0       	  

                      !" !! #$ ## %& %% '( '' )* )) +, ++ -. -- /0 /1 // 23 22 45 46 44 78 77 9: 9; 99 <= <> << ?@ ?A ?B ?? CD CE CF CC GH GG IJ IL KK MN MO MM PQ PR PP ST SU SS VW VX VV YZ Y[ YY \] \\ ^_ ^` ^a ^^ bc bd bb ef ee gh gi gj gg kl km kk no nn pq pr ps pp tu tv tt wy xz xx {| {{ }~ } }} € €
ƒ ‚‚ „
… „„ †‡ †
ˆ †† ‰Š #‹ 
‹ 	Œ   )	Ž G	  „‘ V    	
    { x       "! $# &% (' *) ,+ . 0- 1+ 3 52 6+ 8 :7 ;4 =4 >/ @/ A< B9 D9 E? FC HG JC LK NK OK QM RK TP UP W XS ZV [ ]/ _Y `\ a c^ d f4 hY ie jb lg m o9 qY rn sk up vt y z |{ ~ } x ƒ …‚ ‡„ ˆ  ‚ I KI xw x€ ‚€  ““ ‰ ’’^ ““ ^g ““ gC ““ Cp ““ p? ““ ? ’’ V ““ V” ” ‚	• – 	— 	— 2	— e	— k	— {˜ 	˜ 
	™ 	™ 7	™ n	™ tš K	› › 	› -	› \	› b"
compute_lj_force"
_Z13get_global_idj"
llvm.fmuladd.f32*š
!shoc-1.1.5-MD-compute_lj_force.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

transfer_bytes
€€

devmap_label


wgsize
€
 
transfer_bytes_log1p
}‰A

wgsize_log1p
}‰A