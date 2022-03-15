

[external]
>allocaB4
2
	full_text%
#
!%5 = alloca <4 x float>, align 16
>allocaB4
2
	full_text%
#
!%6 = alloca <4 x float>, align 16
AbitcastB6
4
	full_text'
%
#%7 = bitcast <4 x float>* %5 to i8*
4<4 x float>*B"
 
	full_text

<4 x float>* %5
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %7) #5
"i8*B

	full_text


i8* %7
AbitcastB6
4
	full_text'
%
#%8 = bitcast <4 x float>* %6 to i8*
4<4 x float>*B"
 
	full_text

<4 x float>* %6
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %8) #5
"i8*B

	full_text


i8* %8
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #6
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #6
3sextB+
)
	full_text

%11 = sext i32 %1 to i64
0mulB)
'
	full_text

%12 = mul i64 %10, %11
#i64B

	full_text
	
i64 %10
#i64B

	full_text
	
i64 %11
/addB(
&
	full_text

%13 = add i64 %12, %9
#i64B

	full_text
	
i64 %12
"i64B

	full_text


i64 %9
fgetelementptrBU
S
	full_textF
D
B%14 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %13
#i64B

	full_text
	
i64 %13
WloadBO
M
	full_text@
>
<%15 = load <4 x float>, <4 x float>* %14, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %14
áfsubB
}
	full_textp
n
l%16 = fsub <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %15
3<4 x float>B"
 
	full_text

<4 x float> %15
áfmulB
}
	full_textp
n
l%17 = fmul <4 x float> %16, <float 1.000000e+02, float 1.000000e+02, float 1.000000e+02, float 1.000000e+02>
3<4 x float>B"
 
	full_text

<4 x float> %16
œcallB∆
√
	full_textµ
≤
Ø%18 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> <float 1.000000e+01, float 1.000000e+01, float 1.000000e+01, float 1.000000e+01>, <4 x float> %17)
3<4 x float>B"
 
	full_text

<4 x float> %15
3<4 x float>B"
 
	full_text

<4 x float> %17
áfmulB
}
	full_textp
n
l%19 = fmul <4 x float> %16, <float 1.000000e+01, float 1.000000e+01, float 1.000000e+01, float 1.000000e+01>
3<4 x float>B"
 
	full_text

<4 x float> %16
:faddB2
0
	full_text#
!
%20 = fadd <4 x float> %15, %19
3<4 x float>B"
 
	full_text

<4 x float> %15
3<4 x float>B"
 
	full_text

<4 x float> %19
§fmulBõ
ò
	full_textä
á
Ñ%21 = fmul <4 x float> %16, <float 0x3FA99999A0000000, float 0x3FA99999A0000000, float 0x3FA99999A0000000, float 0x3FA99999A0000000>
3<4 x float>B"
 
	full_text

<4 x float> %16
ÁcallBﬁ
€
	full_textÕ
 
«%22 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>, <4 x float> %21)
3<4 x float>B"
 
	full_text

<4 x float> %15
3<4 x float>B"
 
	full_text

<4 x float> %21
§fmulBõ
ò
	full_textä
á
Ñ%23 = fmul <4 x float> %16, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
3<4 x float>B"
 
	full_text

<4 x float> %16
ÁcallBﬁ
€
	full_textÕ
 
«%24 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>, <4 x float> %23)
3<4 x float>B"
 
	full_text

<4 x float> %15
3<4 x float>B"
 
	full_text

<4 x float> %23
XcallBP
N
	full_textA
?
=%25 = tail call <4 x float> @_Z4sqrtDv4_f(<4 x float> %20) #6
3<4 x float>B"
 
	full_text

<4 x float> %20
:fmulB2
0
	full_text#
!
%26 = fmul <4 x float> %25, %24
3<4 x float>B"
 
	full_text

<4 x float> %25
3<4 x float>B"
 
	full_text

<4 x float> %24
GfdivB?
=
	full_text0
.
,%27 = fdiv <4 x float> %18, %18, !fpmath !12
3<4 x float>B"
 
	full_text

<4 x float> %18
3<4 x float>B"
 
	full_text

<4 x float> %18
WcallBO
M
	full_text@
>
<%28 = tail call <4 x float> @_Z3logDv4_f(<4 x float> %27) #6
3<4 x float>B"
 
	full_text

<4 x float> %27
:fmulB2
0
	full_text#
!
%29 = fmul <4 x float> %24, %24
3<4 x float>B"
 
	full_text

<4 x float> %24
3<4 x float>B"
 
	full_text

<4 x float> %24
ñfdivBç
ä
	full_text}
{
y%30 = fdiv <4 x float> %29, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>, !fpmath !12
3<4 x float>B"
 
	full_text

<4 x float> %29
:faddB2
0
	full_text#
!
%31 = fadd <4 x float> %22, %30
3<4 x float>B"
 
	full_text

<4 x float> %22
3<4 x float>B"
 
	full_text

<4 x float> %30
}callBu
s
	full_textf
d
b%32 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %31, <4 x float> %20, <4 x float> %28)
3<4 x float>B"
 
	full_text

<4 x float> %31
3<4 x float>B"
 
	full_text

<4 x float> %20
3<4 x float>B"
 
	full_text

<4 x float> %28
GfdivB?
=
	full_text0
.
,%33 = fdiv <4 x float> %32, %26, !fpmath !12
3<4 x float>B"
 
	full_text

<4 x float> %32
3<4 x float>B"
 
	full_text

<4 x float> %26
:fsubB2
0
	full_text#
!
%34 = fsub <4 x float> %33, %26
3<4 x float>B"
 
	full_text

<4 x float> %33
3<4 x float>B"
 
	full_text

<4 x float> %26
:fmulB2
0
	full_text#
!
%35 = fmul <4 x float> %22, %20
3<4 x float>B"
 
	full_text

<4 x float> %22
3<4 x float>B"
 
	full_text

<4 x float> %20
çfsubBÑ
Å
	full_textt
r
p%36 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %35
3<4 x float>B"
 
	full_text

<4 x float> %35
WcallBO
M
	full_text@
>
<%37 = tail call <4 x float> @_Z3expDv4_f(<4 x float> %36) #6
3<4 x float>B"
 
	full_text

<4 x float> %36
:fmulB2
0
	full_text#
!
%38 = fmul <4 x float> %18, %37
3<4 x float>B"
 
	full_text

<4 x float> %18
3<4 x float>B"
 
	full_text

<4 x float> %37
VcallBN
L
	full_text?
=
;call void @phi(<4 x float> %33, <4 x float>* nonnull %5) #5
3<4 x float>B"
 
	full_text

<4 x float> %33
4<4 x float>*B"
 
	full_text

<4 x float>* %5
VcallBN
L
	full_text?
=
;call void @phi(<4 x float> %34, <4 x float>* nonnull %6) #5
3<4 x float>B"
 
	full_text

<4 x float> %34
4<4 x float>*B"
 
	full_text

<4 x float>* %6
VloadBN
L
	full_text?
=
;%39 = load <4 x float>, <4 x float>* %5, align 16, !tbaa !9
4<4 x float>*B"
 
	full_text

<4 x float>* %5
VloadBN
L
	full_text?
=
;%40 = load <4 x float>, <4 x float>* %6, align 16, !tbaa !9
4<4 x float>*B"
 
	full_text

<4 x float>* %6
:fmulB2
0
	full_text#
!
%41 = fmul <4 x float> %38, %40
3<4 x float>B"
 
	full_text

<4 x float> %38
3<4 x float>B"
 
	full_text

<4 x float> %40
çfsubBÑ
Å
	full_textt
r
p%42 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %41
3<4 x float>B"
 
	full_text

<4 x float> %41
xcallBp
n
	full_texta
_
]%43 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %18, <4 x float> %39, <4 x float> %42)
3<4 x float>B"
 
	full_text

<4 x float> %18
3<4 x float>B"
 
	full_text

<4 x float> %39
3<4 x float>B"
 
	full_text

<4 x float> %42
fgetelementptrBU
S
	full_textF
D
B%44 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %13
#i64B

	full_text
	
i64 %13
WstoreBN
L
	full_text?
=
;store <4 x float> %43, <4 x float>* %44, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %43
5<4 x float>*B#
!
	full_text

<4 x float>* %44
çfsubBÑ
Å
	full_textt
r
p%45 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %33
3<4 x float>B"
 
	full_text

<4 x float> %33
VcallBN
L
	full_text?
=
;call void @phi(<4 x float> %45, <4 x float>* nonnull %5) #5
3<4 x float>B"
 
	full_text

<4 x float> %45
4<4 x float>*B"
 
	full_text

<4 x float>* %5
çfsubBÑ
Å
	full_textt
r
p%46 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %34
3<4 x float>B"
 
	full_text

<4 x float> %34
VcallBN
L
	full_text?
=
;call void @phi(<4 x float> %46, <4 x float>* nonnull %6) #5
3<4 x float>B"
 
	full_text

<4 x float> %46
4<4 x float>*B"
 
	full_text

<4 x float>* %6
VloadBN
L
	full_text?
=
;%47 = load <4 x float>, <4 x float>* %6, align 16, !tbaa !9
4<4 x float>*B"
 
	full_text

<4 x float>* %6
VloadBN
L
	full_text?
=
;%48 = load <4 x float>, <4 x float>* %5, align 16, !tbaa !9
4<4 x float>*B"
 
	full_text

<4 x float>* %5
:fmulB2
0
	full_text#
!
%49 = fmul <4 x float> %18, %48
3<4 x float>B"
 
	full_text

<4 x float> %18
3<4 x float>B"
 
	full_text

<4 x float> %48
çfsubBÑ
Å
	full_textt
r
p%50 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %49
3<4 x float>B"
 
	full_text

<4 x float> %49
xcallBp
n
	full_texta
_
]%51 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %38, <4 x float> %47, <4 x float> %50)
3<4 x float>B"
 
	full_text

<4 x float> %38
3<4 x float>B"
 
	full_text

<4 x float> %47
3<4 x float>B"
 
	full_text

<4 x float> %50
fgetelementptrBU
S
	full_textF
D
B%52 = getelementptr inbounds <4 x float>, <4 x float>* %3, i64 %13
#i64B

	full_text
	
i64 %13
WstoreBN
L
	full_text?
=
;store <4 x float> %51, <4 x float>* %52, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %51
5<4 x float>*B#
!
	full_text

<4 x float>* %52
WcallBO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %8) #5
"i8*B

	full_text


i8* %8
WcallBO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %7) #5
"i8*B

	full_text


i8* %7
"retB

	full_text


ret void
$i328B

	full_text


i32 %1
6<4 x float>*8B"
 
	full_text

<4 x float>* %2
6<4 x float>*8B"
 
	full_text

<4 x float>* %0
6<4 x float>*8B"
 
	full_text

<4 x float>* %3
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
Ç<4 x float>8Bo
m
	full_text`
^
\<4 x float> <float 1.000000e+01, float 1.000000e+01, float 1.000000e+01, float 1.000000e+01>
$i648B

	full_text


i64 16
Ç<4 x float>8Bo
m
	full_text`
^
\<4 x float> <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
Ü<4 x float>8Bs
q
	full_textd
b
`<4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>
ú<4 x float>8Bà
Ö
	full_textx
v
t<4 x float> <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>
ú<4 x float>8Bà
Ö
	full_textx
v
t<4 x float> <float 0x3FA99999A0000000, float 0x3FA99999A0000000, float 0x3FA99999A0000000, float 0x3FA99999A0000000>
Ç<4 x float>8Bo
m
	full_text`
^
\<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
Ç<4 x float>8Bo
m
	full_text`
^
\<4 x float> <float 1.000000e+02, float 1.000000e+02, float 1.000000e+02, float 1.000000e+02>
#i328B

	full_text	

i32 0
ú<4 x float>8Bà
Ö
	full_textx
v
t<4 x float> <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>        	
 		                        !" !# !! $% $$ &' &( && )* )) +, +- ++ ./ .. 01 02 00 34 35 33 67 66 89 8: 88 ;< ;; => =? == @A @B @C @@ DE DF DD GH GI GG JK JL JJ MN MM OP OO QR QS QQ TU TV TT WX WY WW Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd ce cf cc gh gg ij ik ii lm ll no np nn qr qq st su ss vw vv xy xx z{ z| zz }~ }} Ä 	Å 	Ç  É
Ñ ÉÉ ÖÜ Ö
á ÖÖ à
â àà ä
ã ää åç é gè ê É    
             " # % '$ ( * ,) -! /. 1+ 2 4 53 7+ 9+ :8 <& >; ?= A! B6 C@ E0 FD H0 I& K! LJ NM P RO SD U VG X Y [ ]Q _\ `^ b dZ ea f hc jg kD ml o pG rq t u w y {x |z ~Q Äv Å} Ç Ñ ÜÉ á â ã ëë óó òò ìì íí ïï ññ å îîà òò àä òò ä	 ëë 	6 ïï 6O ññ O íí W óó W ìì @ ìì @c ìì cs óó s ìì + ìì +& ìì & íí n óó nT óó T ëë . îî .ô ô ô 	ö 	ö õ õ 	õ àõ ä	ú ;ù Mù aù lù qù }	û &	û +	ü $† 	° ¢ 	£ )"
blackScholes"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.v4f32"
_Z4sqrtDv4_f"
_Z3logDv4_f"
_Z3expDv4_f"
phi"
llvm.lifetime.end.p0i8*ê
BlackScholes_Kernels.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize_log1p
âboA

devmap_label


wgsize
Ä
 
transfer_bytes_log1p
âboA

transfer_bytes
ÄÄ¿