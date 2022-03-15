

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #4
,addB%
#
	full_text

%7 = add i64 %6, 1
"i64B

	full_text


i64 %6
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
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_local_idj(i32 0) #4
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
6icmpB.
,
	full_text

%11 = icmp slt i32 %8, 1025
"i32B

	full_text


i32 %8
8brB2
0
	full_text#
!
br i1 %11, label %12, label %36
!i1B

	full_text


i1 %11
3srem8B)
'
	full_text

%13 = srem i32 %8, 256
$i328B

	full_text


i32 %8
3mul8B*
(
	full_text

%14 = mul nsw i32 %8, 3
$i328B

	full_text


i32 %8
4srem8B*
(
	full_text

%15 = srem i32 %14, 256
%i328B

	full_text
	
i32 %14
3mul8B*
(
	full_text

%16 = mul nsw i32 %8, 5
$i328B

	full_text


i32 %8
4srem8B*
(
	full_text

%17 = srem i32 %16, 128
%i328B

	full_text
	
i32 %16
5mul8B,
*
	full_text

%18 = mul nsw i32 %17, %4
%i328B

	full_text
	
i32 %17
3add8B*
(
	full_text

%19 = add nsw i32 %3, 1
2add8B)
'
	full_text

%20 = add i32 %18, %15
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %15
2mul8B)
'
	full_text

%21 = mul i32 %20, %19
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %19
6add8B-
+
	full_text

%22 = add nsw i32 %21, %13
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %13
6sext8B,
*
	full_text

%23 = sext i32 %22 to i64
%i328B

	full_text
	
i32 %22
rgetelementptr8B_
]
	full_textP
N
L%24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %23
%i648B

	full_text
	
i64 %23
Kbitcast8B>
<
	full_text/
-
+%25 = bitcast %struct.dcomplex* %24 to i64*
-struct*8B

	full_text

struct* %24
Hload8B>
<
	full_text/
-
+%26 = load i64, i64* %25, align 8, !tbaa !8
'i64*8B

	full_text


i64* %25
0shl8B'
%
	full_text

%27 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
rgetelementptr8B_
]
	full_textP
N
L%29 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %28
%i648B

	full_text
	
i64 %28
Kbitcast8B>
<
	full_text/
-
+%30 = bitcast %struct.dcomplex* %29 to i64*
-struct*8B

	full_text

struct* %29
Hstore8B=
;
	full_text.
,
*store i64 %26, i64* %30, align 8, !tbaa !8
%i648B

	full_text
	
i64 %26
'i64*8B

	full_text


i64* %30
ygetelementptr8Bf
d
	full_textW
U
S%31 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %23, i32 1
%i648B

	full_text
	
i64 %23
Abitcast8B4
2
	full_text%
#
!%32 = bitcast double* %31 to i64*
-double*8B

	full_text

double* %31
Iload8B?
=
	full_text0
.
,%33 = load i64, i64* %32, align 8, !tbaa !13
'i64*8B

	full_text


i64* %32
ygetelementptr8Bf
d
	full_textW
U
S%34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %28, i32 1
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%35 = bitcast double* %34 to i64*
-double*8B

	full_text

double* %34
Istore8B>
<
	full_text/
-
+store i64 %33, i64* %35, align 8, !tbaa !13
%i648B

	full_text
	
i64 %33
'i64*8B

	full_text


i64* %35
'br8B

	full_text

br label %41
0shl8B'
%
	full_text

%37 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%38 = ashr exact i64 %37, 32
%i648B

	full_text
	
i64 %37
ygetelementptr8Bf
d
	full_textW
U
S%39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %38, i32 0
%i648B

	full_text
	
i64 %38
@bitcast8B3
1
	full_text$
"
 %40 = bitcast double* %39 to i8*
-double*8B

	full_text

double* %39
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %40, i8 0, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %40
'br8B

	full_text

br label %41
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Ocall8BE
C
	full_text6
4
2%42 = tail call i64 @_Z14get_local_sizej(i32 0) #4
2lshr8B(
&
	full_text

%43 = lshr i64 %42, 1
%i648B

	full_text
	
i64 %42
8trunc8B-
+
	full_text

%44 = trunc i64 %43 to i32
%i648B

	full_text
	
i64 %43
6icmp8B,
*
	full_text

%45 = icmp sgt i32 %44, 0
%i328B

	full_text
	
i32 %44
:br8B2
0
	full_text#
!
br i1 %45, label %46, label %51
#i18B

	full_text


i1 %45
0shl8B'
%
	full_text

%47 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%48 = ashr exact i64 %47, 32
%i648B

	full_text
	
i64 %47
ygetelementptr8Bf
d
	full_textW
U
S%49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %48, i32 0
%i648B

	full_text
	
i64 %48
ygetelementptr8Bf
d
	full_textW
U
S%50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %48, i32 1
%i648B

	full_text
	
i64 %48
'br8B

	full_text

br label %53
5icmp8B+
)
	full_text

%52 = icmp eq i32 %10, 0
%i328B

	full_text
	
i32 %10
:br8B2
0
	full_text#
!
br i1 %52, label %70, label %81
#i18B

	full_text


i1 %52
Dphi8B;
9
	full_text,
*
(%54 = phi i32 [ %44, %46 ], [ %68, %67 ]
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %68
8icmp8B.
,
	full_text

%55 = icmp sgt i32 %54, %10
%i328B

	full_text
	
i32 %54
%i328B

	full_text
	
i32 %10
:br8B2
0
	full_text#
!
br i1 %55, label %56, label %67
#i18B

	full_text


i1 %55
Nload8BD
B
	full_text5
3
1%57 = load double, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
6add8B-
+
	full_text

%58 = add nsw i32 %54, %10
%i328B

	full_text
	
i32 %54
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%59 = sext i32 %58 to i64
%i328B

	full_text
	
i32 %58
ygetelementptr8Bf
d
	full_textW
U
S%60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %59, i32 0
%i648B

	full_text
	
i64 %59
Nload8BD
B
	full_text5
3
1%61 = load double, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
7fadd8B-
+
	full_text

%62 = fadd double %57, %61
+double8B

	full_text


double %57
+double8B

	full_text


double %61
Oload8BE
C
	full_text6
4
2%63 = load double, double* %50, align 8, !tbaa !13
-double*8B

	full_text

double* %50
ygetelementptr8Bf
d
	full_textW
U
S%64 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 %59, i32 1
%i648B

	full_text
	
i64 %59
Oload8BE
C
	full_text6
4
2%65 = load double, double* %64, align 8, !tbaa !13
-double*8B

	full_text

double* %64
7fadd8B-
+
	full_text

%66 = fadd double %63, %65
+double8B

	full_text


double %63
+double8B

	full_text


double %65
Dstore8B9
7
	full_text*
(
&store double %62, double* %49, align 8
+double8B

	full_text


double %62
-double*8B

	full_text

double* %49
Dstore8B9
7
	full_text*
(
&store double %66, double* %50, align 8
+double8B

	full_text


double %66
-double*8B

	full_text

double* %50
'br8B

	full_text

br label %67
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
2lshr8B(
&
	full_text

%68 = lshr i32 %54, 1
%i328B

	full_text
	
i32 %54
5icmp8B+
)
	full_text

%69 = icmp eq i32 %68, 0
%i328B

	full_text
	
i32 %68
:br8B2
0
	full_text#
!
br i1 %69, label %51, label %53
#i18B

	full_text


i1 %69
Jbitcast8	B=
;
	full_text.
,
*%71 = bitcast %struct.dcomplex* %2 to i64*
Hload8	B>
<
	full_text/
-
+%72 = load i64, i64* %71, align 8, !tbaa !8
'i64*8	B

	full_text


i64* %71
Mcall8	BC
A
	full_text4
2
0%73 = tail call i64 @_Z12get_group_idj(i32 0) #4
rgetelementptr8	B_
]
	full_textP
N
L%74 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %73
%i648	B

	full_text
	
i64 %73
Kbitcast8	B>
<
	full_text/
-
+%75 = bitcast %struct.dcomplex* %74 to i64*
-struct*8	B

	full_text

struct* %74
Hstore8	B=
;
	full_text.
,
*store i64 %72, i64* %75, align 8, !tbaa !8
%i648	B

	full_text
	
i64 %72
'i64*8	B

	full_text


i64* %75
wgetelementptr8	Bd
b
	full_textU
S
Q%76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 0, i32 1
Abitcast8	B4
2
	full_text%
#
!%77 = bitcast double* %76 to i64*
-double*8	B

	full_text

double* %76
Iload8	B?
=
	full_text0
.
,%78 = load i64, i64* %77, align 8, !tbaa !13
'i64*8	B

	full_text


i64* %77
ygetelementptr8	Bf
d
	full_textW
U
S%79 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %73, i32 1
%i648	B

	full_text
	
i64 %73
Abitcast8	B4
2
	full_text%
#
!%80 = bitcast double* %79 to i64*
-double*8	B

	full_text

double* %79
Istore8	B>
<
	full_text/
-
+store i64 %78, i64* %80, align 8, !tbaa !13
%i648	B

	full_text
	
i64 %78
'i64*8	B

	full_text


i64* %80
'br8	B

	full_text

br label %81
$ret8
B

	full_text


ret void
$i328B

	full_text


i32 %4
6struct*8B'
%
	full_text

%struct.dcomplex* %2
6struct*8B'
%
	full_text

%struct.dcomplex* %0
6struct*8B'
%
	full_text

%struct.dcomplex* %1
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
#i328B

	full_text	

i32 3
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
$i648B

	full_text


i64 16
%i18B

	full_text


i1 false
%i328B

	full_text
	
i32 256
!i88B

	full_text

i8 0
&i328B

	full_text


i32 1025
#i328B

	full_text	

i32 5
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
%i328B

	full_text
	
i32 128        	
 		                      !  "    #$ ## %& %% '( '' )* )) +, ++ -. -- /0 // 12 11 34 35 33 67 66 89 88 :; :: <= << >? >> @A @B @@ CE DD FG FF HI HH JK JJ LM LL NO PP QR QQ ST SS UV UU WX WZ YY [\ [[ ]^ ]] _` __ ac bb de dg fh ff ij ik ii lm lo nn pq pr pp st ss uv uu wx ww yz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå çé çç èê èè ëí ëì îï îî ññ ó
ò óó ôö ôô õú õ
ù õõ ûû ü† üü °¢ °° £
§ ££ •¶ •• ß® ß
© ßß ™	¨ ≠ /≠ <≠ H≠ ]≠ _≠ u≠ ~≠ ì≠ ûÆ %Æ 6Ø óØ £∞     
	            ! "  $# &% (' * ,+ .- 0/ 2) 41 5# 76 98 ;- =< ?: A> B ED GF IH KJ MP RQ TS VU X ZY \[ ^[ ` cb eS gç hf j ki m] of q rp ts vu xn zw {_ }s ~ Å| ÉÄ Ñy Ü] áÇ â_ äf éç êè íì ïñ òó öî úô ùû †ü ¢ñ §£ ¶° ®• ©  DC ON OW YW ba fd ìd ´l nl å™ ´ã åë bë f ´ ∂∂ ≤≤ ±± ¥¥ µµ ≥≥L ∂∂ L ≤≤ O ≥≥ OP ¥¥ P ±± å ≥≥ åñ µµ ñ	∑ ∏ ∏ 	∏ H∏ P	∏ U	∏ ]	∏ b	∏ u
∏ è∏ ñ	π 	π 6	π <π O	π _	π ~π å
π ç
π û
π £	∫ +	∫ -	∫ D	∫ F	∫ Y	∫ [	ª L	º L	Ω 	Ω 	æ L	ø 		¿ 	¡ 	¡ Q
¬ û	√ "

checksum"
_Z13get_global_idj"
_Z12get_local_idj"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.memset.p0i8.i64*ã
npb-FT-checksum.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize
 

devmap_label
 
 
transfer_bytes_log1p
€ùA

transfer_bytes	
ê¥–†

wgsize_log1p
€ùA