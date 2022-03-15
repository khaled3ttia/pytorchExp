

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
McallBE
C
	full_text6
4
2%7 = tail call i64 @_Z15get_global_sizej(i32 0) #2
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
3icmpB+
)
	full_text

%9 = icmp slt i32 %6, %8
"i32B

	full_text


i32 %6
"i32B

	full_text


i32 %8
3icmpB+
)
	full_text

%10 = icmp sgt i32 %6, 0
"i32B

	full_text


i32 %6
.andB'
%
	full_text

%11 = and i1 %9, %10
 i1B

	full_text	

i1 %9
!i1B

	full_text


i1 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %99
!i1B

	full_text


i1 %11
8and8B/
-
	full_text 

%13 = and i64 %5, 4294967295
$i648B

	full_text


i64 %5
5add8B,
*
	full_text

%14 = add nsw i64 %13, -1
%i648B

	full_text
	
i64 %13
/and8B&
$
	full_text

%15 = and i64 %5, 7
$i648B

	full_text


i64 %5
6icmp8B,
*
	full_text

%16 = icmp ult i64 %14, 7
%i648B

	full_text
	
i64 %14
:br8B2
0
	full_text#
!
br i1 %16, label %57, label %17
#i18B

	full_text


i1 %16
6sub8B-
+
	full_text

%18 = sub nsw i64 %13, %15
%i648B

	full_text
	
i64 %13
%i648B

	full_text
	
i64 %15
'br8B

	full_text

br label %19
Bphi8B9
7
	full_text*
(
&%20 = phi i64 [ 0, %17 ], [ %54, %19 ]
%i648B

	full_text
	
i64 %54
Bphi8B9
7
	full_text*
(
&%21 = phi i32 [ 0, %17 ], [ %53, %19 ]
%i328B

	full_text
	
i32 %53
Dphi8B;
9
	full_text,
*
(%22 = phi i64 [ %18, %17 ], [ %55, %19 ]
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %55
Xgetelementptr8BE
C
	full_text6
4
2%23 = getelementptr inbounds i32, i32* %1, i64 %20
%i648B

	full_text
	
i64 %20
Hload8B>
<
	full_text/
-
+%24 = load i32, i32* %23, align 4, !tbaa !8
'i32*8B

	full_text


i32* %23
6add8B-
+
	full_text

%25 = add nsw i32 %24, %21
%i328B

	full_text
	
i32 %24
%i328B

	full_text
	
i32 %21
.or8B&
$
	full_text

%26 = or i64 %20, 1
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%27 = getelementptr inbounds i32, i32* %1, i64 %26
%i648B

	full_text
	
i64 %26
Hload8B>
<
	full_text/
-
+%28 = load i32, i32* %27, align 4, !tbaa !8
'i32*8B

	full_text


i32* %27
6add8B-
+
	full_text

%29 = add nsw i32 %28, %25
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %25
.or8B&
$
	full_text

%30 = or i64 %20, 2
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%31 = getelementptr inbounds i32, i32* %1, i64 %30
%i648B

	full_text
	
i64 %30
Hload8B>
<
	full_text/
-
+%32 = load i32, i32* %31, align 4, !tbaa !8
'i32*8B

	full_text


i32* %31
6add8B-
+
	full_text

%33 = add nsw i32 %32, %29
%i328B

	full_text
	
i32 %32
%i328B

	full_text
	
i32 %29
.or8B&
$
	full_text

%34 = or i64 %20, 3
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%35 = getelementptr inbounds i32, i32* %1, i64 %34
%i648B

	full_text
	
i64 %34
Hload8B>
<
	full_text/
-
+%36 = load i32, i32* %35, align 4, !tbaa !8
'i32*8B

	full_text


i32* %35
6add8B-
+
	full_text

%37 = add nsw i32 %36, %33
%i328B

	full_text
	
i32 %36
%i328B

	full_text
	
i32 %33
.or8B&
$
	full_text

%38 = or i64 %20, 4
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%39 = getelementptr inbounds i32, i32* %1, i64 %38
%i648B

	full_text
	
i64 %38
Hload8B>
<
	full_text/
-
+%40 = load i32, i32* %39, align 4, !tbaa !8
'i32*8B

	full_text


i32* %39
6add8B-
+
	full_text

%41 = add nsw i32 %40, %37
%i328B

	full_text
	
i32 %40
%i328B

	full_text
	
i32 %37
.or8B&
$
	full_text

%42 = or i64 %20, 5
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%43 = getelementptr inbounds i32, i32* %1, i64 %42
%i648B

	full_text
	
i64 %42
Hload8B>
<
	full_text/
-
+%44 = load i32, i32* %43, align 4, !tbaa !8
'i32*8B

	full_text


i32* %43
6add8B-
+
	full_text

%45 = add nsw i32 %44, %41
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %41
.or8B&
$
	full_text

%46 = or i64 %20, 6
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%47 = getelementptr inbounds i32, i32* %1, i64 %46
%i648B

	full_text
	
i64 %46
Hload8B>
<
	full_text/
-
+%48 = load i32, i32* %47, align 4, !tbaa !8
'i32*8B

	full_text


i32* %47
6add8B-
+
	full_text

%49 = add nsw i32 %48, %45
%i328B

	full_text
	
i32 %48
%i328B

	full_text
	
i32 %45
.or8B&
$
	full_text

%50 = or i64 %20, 7
%i648B

	full_text
	
i64 %20
Xgetelementptr8BE
C
	full_text6
4
2%51 = getelementptr inbounds i32, i32* %1, i64 %50
%i648B

	full_text
	
i64 %50
Hload8B>
<
	full_text/
-
+%52 = load i32, i32* %51, align 4, !tbaa !8
'i32*8B

	full_text


i32* %51
6add8B-
+
	full_text

%53 = add nsw i32 %52, %49
%i328B

	full_text
	
i32 %52
%i328B

	full_text
	
i32 %49
4add8B+
)
	full_text

%54 = add nsw i64 %20, 8
%i648B

	full_text
	
i64 %20
1add8B(
&
	full_text

%55 = add i64 %22, -8
%i648B

	full_text
	
i64 %22
5icmp8B+
)
	full_text

%56 = icmp eq i64 %55, 0
%i648B

	full_text
	
i64 %55
:br8B2
0
	full_text#
!
br i1 %56, label %57, label %19
#i18B

	full_text


i1 %56
Fphi8B=
;
	full_text.
,
*%58 = phi i32 [ undef, %12 ], [ %53, %19 ]
%i328B

	full_text
	
i32 %53
Bphi8B9
7
	full_text*
(
&%59 = phi i64 [ 0, %12 ], [ %54, %19 ]
%i648B

	full_text
	
i64 %54
Bphi8B9
7
	full_text*
(
&%60 = phi i32 [ 0, %12 ], [ %53, %19 ]
%i328B

	full_text
	
i32 %53
5icmp8B+
)
	full_text

%61 = icmp eq i64 %15, 0
%i648B

	full_text
	
i64 %15
:br8B2
0
	full_text#
!
br i1 %61, label %73, label %62
#i18B

	full_text


i1 %61
'br8B

	full_text

br label %63
Dphi8B;
9
	full_text,
*
(%64 = phi i64 [ %59, %62 ], [ %70, %63 ]
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %70
Dphi8B;
9
	full_text,
*
(%65 = phi i32 [ %60, %62 ], [ %69, %63 ]
%i328B

	full_text
	
i32 %60
%i328B

	full_text
	
i32 %69
Dphi8B;
9
	full_text,
*
(%66 = phi i64 [ %15, %62 ], [ %71, %63 ]
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %71
Xgetelementptr8BE
C
	full_text6
4
2%67 = getelementptr inbounds i32, i32* %1, i64 %64
%i648B

	full_text
	
i64 %64
Hload8B>
<
	full_text/
-
+%68 = load i32, i32* %67, align 4, !tbaa !8
'i32*8B

	full_text


i32* %67
6add8B-
+
	full_text

%69 = add nsw i32 %68, %65
%i328B

	full_text
	
i32 %68
%i328B

	full_text
	
i32 %65
8add8B/
-
	full_text 

%70 = add nuw nsw i64 %64, 1
%i648B

	full_text
	
i64 %64
1add8B(
&
	full_text

%71 = add i64 %66, -1
%i648B

	full_text
	
i64 %66
5icmp8B+
)
	full_text

%72 = icmp eq i64 %71, 0
%i648B

	full_text
	
i64 %71
Jbr8BB
@
	full_text3
1
/br i1 %72, label %73, label %63, !llvm.loop !12
#i18B

	full_text


i1 %72
Dphi8B;
9
	full_text,
*
(%74 = phi i32 [ %58, %57 ], [ %69, %63 ]
%i328B

	full_text
	
i32 %58
%i328B

	full_text
	
i32 %69
6icmp8B,
*
	full_text

%75 = icmp sgt i32 %74, 0
%i328B

	full_text
	
i32 %74
:br8B2
0
	full_text#
!
br i1 %75, label %76, label %99
#i18B

	full_text


i1 %75
4icmp8B*
(
	full_text

%77 = icmp eq i32 %6, 0
$i328B

	full_text


i32 %6
0shl8B'
%
	full_text

%78 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%79 = ashr exact i64 %78, 32
%i648B

	full_text
	
i64 %78
:br8B2
0
	full_text#
!
br i1 %77, label %84, label %80
#i18B

	full_text


i1 %77
Xgetelementptr8	BE
C
	full_text6
4
2%81 = getelementptr inbounds i32, i32* %2, i64 %79
%i648	B

	full_text
	
i64 %79
Hload8	B>
<
	full_text/
-
+%82 = load i32, i32* %81, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %81
4add8	B+
)
	full_text

%83 = add nsw i32 %82, 1
%i328	B

	full_text
	
i32 %82
'br8	B

	full_text

br label %84
Bphi8
B9
7
	full_text*
(
&%85 = phi i32 [ %83, %80 ], [ 0, %76 ]
%i328
B

	full_text
	
i32 %83
Xgetelementptr8
BE
C
	full_text6
4
2%86 = getelementptr inbounds i32, i32* %3, i64 %79
%i648
B

	full_text
	
i64 %79
Hload8
B>
<
	full_text/
-
+%87 = load i32, i32* %86, align 4, !tbaa !8
'i32*8
B

	full_text


i32* %86
8icmp8
B.
,
	full_text

%88 = icmp sgt i32 %85, %87
%i328
B

	full_text
	
i32 %85
%i328
B

	full_text
	
i32 %87
:br8
B2
0
	full_text#
!
br i1 %88, label %99, label %89
#i18
B

	full_text


i1 %88
6sext8B,
*
	full_text

%90 = sext i32 %85 to i64
%i328B

	full_text
	
i32 %85
6sext8B,
*
	full_text

%91 = sext i32 %87 to i64
%i328B

	full_text
	
i32 %87
'br8B

	full_text

br label %92
Dphi8B;
9
	full_text,
*
(%93 = phi i64 [ %97, %92 ], [ %90, %89 ]
%i648B

	full_text
	
i64 %97
%i648B

	full_text
	
i64 %90
Xgetelementptr8BE
C
	full_text6
4
2%94 = getelementptr inbounds i32, i32* %0, i64 %93
%i648B

	full_text
	
i64 %93
Hload8B>
<
	full_text/
-
+%95 = load i32, i32* %94, align 4, !tbaa !8
'i32*8B

	full_text


i32* %94
6add8B-
+
	full_text

%96 = add nsw i32 %95, %74
%i328B

	full_text
	
i32 %95
%i328B

	full_text
	
i32 %74
Hstore8B=
;
	full_text.
,
*store i32 %96, i32* %94, align 4, !tbaa !8
%i328B

	full_text
	
i32 %96
'i32*8B

	full_text


i32* %94
4add8B+
)
	full_text

%97 = add nsw i64 %93, 1
%i648B

	full_text
	
i64 %93
8icmp8B.
,
	full_text

%98 = icmp slt i64 %93, %91
%i648B

	full_text
	
i64 %93
%i648B

	full_text
	
i64 %91
:br8B2
0
	full_text#
!
br i1 %98, label %92, label %99
#i18B

	full_text


i1 %98
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %3
&i32*8B

	full_text
	
i32* %2
&i32*8B

	full_text
	
i32* %1
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
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -8
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 8
'i328B

	full_text

	i32 undef
#i648B

	full_text	

i64 4
,i648B!

	full_text

i64 4294967295
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 6
#i648B

	full_text	

i64 5       	  
 

                    !" !! #$ #% ## &' && () (( *+ *, ** -. -- /0 // 12 11 34 35 33 67 66 89 88 :; :: <= <> << ?@ ?? AB AA CD CC EF EG EE HI HH JK JJ LM LL NO NP NN QR QQ ST SS UV UU WX WY WW Z[ ZZ \] \\ ^_ ^^ `a `b `` cd cc ef ee gh gg ij ik ii lm ll no nn pq pp rs ru tt vw vv xy xx z{ zz |} |€ 	  ‚ƒ ‚
„ ‚‚ …† …
‡ …… 
‰  ‹   
    ‘’ ‘‘ “” ““ •– • —
™ —— ›      ΅    Ά£ ΆΆ ¤¥ ¤
§ ¦¦ ¨© ¨¨ ª« ªª ¬® ­­ ―
° ―― ±² ±± ³΄ ³
µ ³³ ¶· ¶Ή ΈΈ Ί» ΊΊ ΌΎ ½
Ώ ½½ ΐ
Α ΐΐ ΒΓ ΒΒ ΔΕ Δ
Ζ ΔΔ ΗΘ Η
Ι ΗΗ ΚΛ ΚΚ ΜΝ Μ
Ξ ΜΜ ΟΠ ΟÒ ΐΣ ―Τ ¦Υ &Υ /Υ 8Υ AΥ JΥ SΥ \Υ eΥ     	  
         l  i " $n % '& )( +! , .- 0/ 21 4* 5 76 98 ;: =3 > @? BA DC F< G IH KJ ML OE P RQ TS VU XN Y [Z ]\ _^ aW b dc fe hg j` k m# on qp si ul wi y {z }v € x ƒ „ †‘ ‡ ‰ ‹ ‚  … ’‘ ”“ –t  ™— ›   ΅  £ ¥Ά §¦ ©¨ «ª ®Ά °― ²­ ΄± µ³ ·­ Ή± »Κ ΎΈ Ώ½ Αΐ ΓΒ Ε— ΖΔ Θΐ Ι½ Λ½ ΝΊ ΞΜ Π  Ρ t | —| ~   Ρ~ r tr ¤ ­¤ ¦• —• ¶ Ρ¶ Έ¬ ­Ό ½Ο ½Ο Ρ Ρ ΦΦ ΧΧ ΦΦ  ΧΧ Ψ 	Ψ pΨ v	Ψ z
Ψ “	Ω n
Ϊ  
Ϊ Ά	Ϋ 6ά ά 	ά 
ά !ά x
ά 
ά 
ά ­	έ lή t	ί H	ΰ 
α ª	β -
β 
β Κ	γ 	γ 	γ c	δ 
δ ‘	ε ?	ζ Z	η Q"	
makea_2"
_Z13get_global_idj"
_Z15get_global_sizej*
npb-CG-makea_2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02€

transfer_bytes
όδƒ

devmap_label
 

wgsize
 
 
transfer_bytes_log1p
½„A

wgsize_log1p
½„A