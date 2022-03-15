

[external]
?allocaB5
3
	full_text&
$
"%6 = alloca [5 x double], align 16
?allocaB5
3
	full_text&
$
"%7 = alloca [5 x double], align 16
BbitcastB7
5
	full_text(
&
$%8 = bitcast [5 x double]* %6 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %6
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %8) #4
"i8*B

	full_text


i8* %8
BbitcastB7
5
	full_text(
&
$%9 = bitcast [5 x double]* %7 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %7
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %9) #4
"i8*B

	full_text


i8* %9
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #5
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 0) #5
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
5icmpB-
+
	full_text

%14 = icmp slt i32 %11, %4
#i32B

	full_text
	
i32 %11
5icmpB-
+
	full_text

%15 = icmp slt i32 %13, %2
#i32B

	full_text
	
i32 %13
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
br i1 %16, label %17, label %75
!i1B

	full_text


i1 %16
Wbitcast8BJ
H
	full_text;
9
7%18 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
ogetelementptr8B\
Z
	full_textM
K
I%19 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
jcall8B`
^
	full_textQ
O
Mcall void @exact(i32 %13, i32 0, i32 %11, double* nonnull %19, double* %1) #4
%i328B

	full_text
	
i32 %13
%i328B

	full_text
	
i32 %11
-double*8B

	full_text

double* %19
4add8B+
)
	full_text

%20 = add nsw i32 %3, -1
ogetelementptr8B\
Z
	full_textM
K
I%21 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
lcall8Bb
`
	full_textS
Q
Ocall void @exact(i32 %13, i32 %20, i32 %11, double* nonnull %21, double* %1) #4
%i328B

	full_text
	
i32 %13
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %11
-double*8B

	full_text

double* %21
1shl8B(
&
	full_text

%22 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
1shl8B(
&
	full_text

%24 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
6sext8B,
*
	full_text

%26 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
Fbitcast8B9
7
	full_text*
(
&%27 = bitcast [5 x double]* %6 to i64*
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Iload8B?
=
	full_text0
.
,%28 = load i64, i64* %27, align 16, !tbaa !8
'i64*8B

	full_text


i64* %27
†getelementptr8Bå
â
	full_text|
z
x%29 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 0, i64 %25, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
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
Fbitcast8B9
7
	full_text*
(
&%31 = bitcast [5 x double]* %7 to i64*
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Iload8B?
=
	full_text0
.
,%32 = load i64, i64* %31, align 16, !tbaa !8
'i64*8B

	full_text


i64* %31
¢getelementptr8Bé
ã
	full_text~
|
z%33 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 %26, i64 %25, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%34 = bitcast double* %33 to i64*
-double*8B

	full_text

double* %33
Hstore8B=
;
	full_text.
,
*store i64 %32, i64* %34, align 8, !tbaa !8
%i648B

	full_text
	
i64 %32
'i64*8B

	full_text


i64* %34
ogetelementptr8B\
Z
	full_textM
K
I%35 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Abitcast8B4
2
	full_text%
#
!%36 = bitcast double* %35 to i64*
-double*8B

	full_text

double* %35
Hload8B>
<
	full_text/
-
+%37 = load i64, i64* %36, align 8, !tbaa !8
'i64*8B

	full_text


i64* %36
†getelementptr8Bå
â
	full_text|
z
x%38 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 0, i64 %25, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
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
!%39 = bitcast double* %38 to i64*
-double*8B

	full_text

double* %38
Hstore8B=
;
	full_text.
,
*store i64 %37, i64* %39, align 8, !tbaa !8
%i648B

	full_text
	
i64 %37
'i64*8B

	full_text


i64* %39
ogetelementptr8B\
Z
	full_textM
K
I%40 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Abitcast8B4
2
	full_text%
#
!%41 = bitcast double* %40 to i64*
-double*8B

	full_text

double* %40
Hload8B>
<
	full_text/
-
+%42 = load i64, i64* %41, align 8, !tbaa !8
'i64*8B

	full_text


i64* %41
¢getelementptr8Bé
ã
	full_text~
|
z%43 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 %26, i64 %25, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%44 = bitcast double* %43 to i64*
-double*8B

	full_text

double* %43
Hstore8B=
;
	full_text.
,
*store i64 %42, i64* %44, align 8, !tbaa !8
%i648B

	full_text
	
i64 %42
'i64*8B

	full_text


i64* %44
ogetelementptr8B\
Z
	full_textM
K
I%45 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Abitcast8B4
2
	full_text%
#
!%46 = bitcast double* %45 to i64*
-double*8B

	full_text

double* %45
Iload8B?
=
	full_text0
.
,%47 = load i64, i64* %46, align 16, !tbaa !8
'i64*8B

	full_text


i64* %46
†getelementptr8Bå
â
	full_text|
z
x%48 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 0, i64 %25, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
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
!%49 = bitcast double* %48 to i64*
-double*8B

	full_text

double* %48
Hstore8B=
;
	full_text.
,
*store i64 %47, i64* %49, align 8, !tbaa !8
%i648B

	full_text
	
i64 %47
'i64*8B

	full_text


i64* %49
ogetelementptr8B\
Z
	full_textM
K
I%50 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Abitcast8B4
2
	full_text%
#
!%51 = bitcast double* %50 to i64*
-double*8B

	full_text

double* %50
Iload8B?
=
	full_text0
.
,%52 = load i64, i64* %51, align 16, !tbaa !8
'i64*8B

	full_text


i64* %51
¢getelementptr8Bé
ã
	full_text~
|
z%53 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 %26, i64 %25, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%54 = bitcast double* %53 to i64*
-double*8B

	full_text

double* %53
Hstore8B=
;
	full_text.
,
*store i64 %52, i64* %54, align 8, !tbaa !8
%i648B

	full_text
	
i64 %52
'i64*8B

	full_text


i64* %54
ogetelementptr8B\
Z
	full_textM
K
I%55 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Abitcast8B4
2
	full_text%
#
!%56 = bitcast double* %55 to i64*
-double*8B

	full_text

double* %55
Hload8B>
<
	full_text/
-
+%57 = load i64, i64* %56, align 8, !tbaa !8
'i64*8B

	full_text


i64* %56
†getelementptr8Bå
â
	full_text|
z
x%58 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 0, i64 %25, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
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
!%59 = bitcast double* %58 to i64*
-double*8B

	full_text

double* %58
Hstore8B=
;
	full_text.
,
*store i64 %57, i64* %59, align 8, !tbaa !8
%i648B

	full_text
	
i64 %57
'i64*8B

	full_text


i64* %59
ogetelementptr8B\
Z
	full_textM
K
I%60 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Abitcast8B4
2
	full_text%
#
!%61 = bitcast double* %60 to i64*
-double*8B

	full_text

double* %60
Hload8B>
<
	full_text/
-
+%62 = load i64, i64* %61, align 8, !tbaa !8
'i64*8B

	full_text


i64* %61
¢getelementptr8Bé
ã
	full_text~
|
z%63 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 %26, i64 %25, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%64 = bitcast double* %63 to i64*
-double*8B

	full_text

double* %63
Hstore8B=
;
	full_text.
,
*store i64 %62, i64* %64, align 8, !tbaa !8
%i648B

	full_text
	
i64 %62
'i64*8B

	full_text


i64* %64
ogetelementptr8B\
Z
	full_textM
K
I%65 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Abitcast8B4
2
	full_text%
#
!%66 = bitcast double* %65 to i64*
-double*8B

	full_text

double* %65
Iload8B?
=
	full_text0
.
,%67 = load i64, i64* %66, align 16, !tbaa !8
'i64*8B

	full_text


i64* %66
†getelementptr8Bå
â
	full_text|
z
x%68 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 0, i64 %25, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
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
!%69 = bitcast double* %68 to i64*
-double*8B

	full_text

double* %68
Hstore8B=
;
	full_text.
,
*store i64 %67, i64* %69, align 8, !tbaa !8
%i648B

	full_text
	
i64 %67
'i64*8B

	full_text


i64* %69
ogetelementptr8B\
Z
	full_textM
K
I%70 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Abitcast8B4
2
	full_text%
#
!%71 = bitcast double* %70 to i64*
-double*8B

	full_text

double* %70
Iload8B?
=
	full_text0
.
,%72 = load i64, i64* %71, align 16, !tbaa !8
'i64*8B

	full_text


i64* %71
¢getelementptr8Bé
ã
	full_text~
|
z%73 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %18, i64 %23, i64 %26, i64 %25, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%74 = bitcast double* %73 to i64*
-double*8B

	full_text

double* %73
Hstore8B=
;
	full_text.
,
*store i64 %72, i64* %74, align 8, !tbaa !8
%i648B

	full_text
	
i64 %72
'i64*8B

	full_text


i64* %74
'br8B

	full_text

br label %75
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %9) #4
$i8*8B

	full_text


i8* %9
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %8) #4
$i8*8B

	full_text


i8* %8
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %3
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
$i648B

	full_text


i64 40
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 32
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 0        	
 		                       !! "# "" $% $& $' $( $$ )* )) +, ++ -. -- /0 // 12 11 34 33 56 55 78 79 7: 77 ;< ;; => =? == @A @@ BC BB DE DF DG DH DD IJ II KL KM KK NO NN PQ PP RS RR TU TV TW TT XY XX Z[ Z\ ZZ ]^ ]] _` __ ab aa cd ce cf cg cc hi hh jk jl jj mn mm op oo qr qq st su sv ss wx ww yz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà áá âä â
ã ââ åç åå éè éé êë êê íì í
î í
ï íí ñó ññ òô ò
ö òò õú õõ ùû ùù ü† üü °¢ °
£ °
§ °
• °° ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿
√ ¿
ƒ ¿¿ ≈∆ ≈≈ «» «
… ««  
Ã ÀÀ Õ
Œ ÕÕ œ– 	— 	“ 	“ $” !	‘     
             # %! & '" ( *) , .- 0! 2 43 6 8+ 9/ :7 <5 >; ? A@ C E+ F1 G/ HD JB LI M ON QP S U+ V/ WT YR [X \ ^] `_ b d+ e1 f/ gc ia kh l nm po r t+ u/ vs xq zw { }| ~ Å É+ Ñ1 Ö/ ÜÇ àÄ äá ã çå èé ë ì+ î/ ïí óê ôñ ö úõ ûù † ¢+ £1 §/ •° ßü ©¶ ™ ¨´ Æ≠ ∞ ≤+ ≥/ ¥± ∂Ø ∏µ π ª∫ Ωº ø ¡+ ¬1 √/ ƒ¿ ∆æ »≈ … Ã Œ  À  À ◊◊ ÿÿ ’’ ÷÷ œÀ ÿÿ À ÷÷  ’’ $ ◊◊ $	 ’’ 	 ÷÷  ◊◊ Õ ÿÿ ÕŸ Ÿ 	Ÿ ÀŸ Õ	⁄ N	⁄ T	⁄ ]	⁄ c
€ å
€ í
€ õ
€ °‹ 	‹ › › › 	ﬁ m	ﬁ s	ﬁ |
ﬁ Ç	ﬂ )	ﬂ +	ﬂ -	ﬂ /	‡ !
· ´
· ±
· ∫
· ¿	‚ 	‚ 	‚ "	‚ "	‚ 7	‚ 7	‚ D	‚ N	‚ T	‚ ]	‚ m	‚ s	‚ |
‚ å
‚ í
‚ õ
‚ ´
‚ ±
‚ ∫"
setbv2"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
exact"
llvm.lifetime.end.p0i8*â
npb-LU-setbv2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä
 
transfer_bytes_log1p
⁄açA

wgsize
<

transfer_bytes
òì…

devmap_label


wgsize_log1p
⁄açA